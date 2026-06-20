//! augur — "predict, don't pack".
//!
//! A from-scratch context-mixing compressor. The whole engine is one idea:
//! predict the next bit, code only the surprise. Every model is a `predict()`
//! that returns P(next bit = 1); a logistic mixer blends them; one binary
//! arithmetic coder turns the blended probability into bits. The encoder and
//! decoder run the *identical* predict -> code -> update loop, so they can never
//! drift out of sync (the classic context-mixing failure mode).
//!
//! Portfolio:
//!   - order 0..4 direct context models (local statistics)
//!   - a match model (long-range repeats — what byte contexts can't see)
//!   - STRUCTURE models: a streaming JSON parser exposes "which field's value am
//!     I inside"; we condition on (field, position) and (field, depth).
//!   - NUMERIC model: per-field linear extrapolation. For each field it tracks
//!     last value + delta and predicts the digits of `last + delta` before they
//!     are read. Auto-increment IDs, timestamps, counters collapse to near-zero.
//!     This is "the match model, but the source is a formula instead of history".
//!
//! The parser/numeric models are deterministic heuristics (not validators), so
//! they cannot desync and degrade to ignorable extra inputs on non-matching data
//! (the mixer simply learns to down-weight them).

use std::env;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Binary arithmetic coder (carryless, 32-bit). p is P(bit==1) in 12-bit units.
// ---------------------------------------------------------------------------

struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { x1: 0, x2: 0xffff_ffff, out: Vec::new() }
    }

    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
        let range = (self.x2 - self.x1) as u64;
        let xmid = self.x1 + ((range * p as u64) >> 12) as u32;
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
        self.out
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    inp: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Self { x1: 0, x2: 0xffff_ffff, x: 0, inp, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next_byte() as u32;
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = if self.pos < self.inp.len() { self.inp[self.pos] } else { 0 };
        self.pos += 1;
        b
    }

    #[inline]
    fn decode(&mut self, p: u32) -> u32 {
        let range = (self.x2 - self.x1) as u64;
        let xmid = self.x1 + ((range * p as u64) >> 12) as u32;
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
            self.x = (self.x << 8) | self.next_byte() as u32;
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// Predictor: portfolio of models + logistic mixer.
// ---------------------------------------------------------------------------

const MEM_BITS: usize = 22;
const MASK: usize = (1 << MEM_BITS) - 1;
const NORD: usize = 5; // byte-context orders 0..=4
const NSTR: usize = 2; // structure-aware models
const NTAB: usize = NORD + NSTR; // table-backed models
const NIN: usize = NTAB + 2; // + match model + numeric model
const MINLEN: usize = 6; // match model min context length
const RATE: i32 = 4; // context table adaptation rate
const LR: f64 = 0.02; // mixer learning rate
const ARRAY_TAG: u32 = 0xA22A_5151; // field tag for array elements
const NUMSLOTS: usize = 1 << 16; // per-field numeric state slots

#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[inline]
fn squash(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn hstep(h: u32, c: u8) -> u32 {
    (h ^ c as u32).wrapping_mul(0x0100_0193)
}

#[inline]
fn hash_n(buf: &[u8], n: usize, k: usize) -> u32 {
    let mut h = 0x811c_9dc5u32 ^ (k as u32).wrapping_mul(0x9e37_79b1);
    for j in (n - k)..n {
        h = hstep(h, buf[j]);
    }
    h
}

#[derive(Clone, Copy)]
struct Frame {
    is_object: bool,
    key_hash: u32,
    expect_key: bool,
}

#[derive(Clone, Copy, Default)]
struct NumState {
    last: i64,
    delta: i64,
    hits: u32,
    seen: bool,
}

struct Predictor {
    buf: Vec<u8>,
    t: Vec<Vec<u16>>,
    mm: Vec<u32>,
    // bit-assembly state
    c0: u32,
    bitpos: u32,
    ctxh: [u32; NTAB],
    // match model state
    mm_on: bool,
    match_ptr: usize,
    match_len: u32,
    pb: u8,
    // streaming JSON parser state
    in_str: bool,
    esc: bool,
    str_is_key: bool,
    cur_str_hash: u32,
    stack: Vec<Frame>,
    vpos: u32,
    value_pending: bool,
    // numeric model state
    num: Vec<NumState>,
    in_num_value: bool,
    cur_num: i64,
    cur_len: u32,
    cur_neg: bool,
    cur_is_num: bool,
    cur_field: u32,
    np_digits: [u8; 24],
    np_len: usize,
    np_ptr: usize,
    np_active: bool,
    np_conf: f64,
    // mixer
    w: Vec<[f64; NIN]>,
    // cached for update()
    idx: [usize; NTAB],
    st: [f64; NIN],
    wsel: usize,
    p1: f64,
}

impl Predictor {
    fn new() -> Self {
        let mut p = Self {
            buf: Vec::new(),
            t: vec![vec![2048u16; 1 << MEM_BITS]; NTAB],
            mm: vec![0u32; 1 << MEM_BITS],
            c0: 1,
            bitpos: 0,
            ctxh: [0; NTAB],
            mm_on: false,
            match_ptr: 0,
            match_len: 0,
            pb: 0,
            in_str: false,
            esc: false,
            str_is_key: false,
            cur_str_hash: 0,
            stack: Vec::with_capacity(32),
            vpos: 0,
            value_pending: false,
            num: vec![NumState::default(); NUMSLOTS],
            in_num_value: false,
            cur_num: 0,
            cur_len: 0,
            cur_neg: false,
            cur_is_num: false,
            cur_field: 0,
            np_digits: [0; 24],
            np_len: 0,
            np_ptr: 0,
            np_active: false,
            np_conf: 0.0,
            w: vec![[0.15; NIN]; 256],
            idx: [0; NTAB],
            st: [0.0; NIN],
            wsel: 0,
            p1: 0.5,
        };
        p.recompute_ctx();
        p
    }

    #[inline]
    fn field_hash(&self) -> u32 {
        match self.stack.last() {
            Some(f) if f.is_object => f.key_hash,
            Some(_) => ARRAY_TAG,
            None => 0,
        }
    }

    fn recompute_ctx(&mut self) {
        let n = self.buf.len();
        self.ctxh[0] = 0x1234_5678;
        for k in 1..NORD {
            self.ctxh[k] = if n >= k {
                hash_n(&self.buf, n, k)
            } else {
                (k as u32).wrapping_mul(0x9e37_79b1)
            };
        }
        let field = self.field_hash();
        let depth = self.stack.len().min(15) as u32;
        let in_val_str = (self.in_str && !self.str_is_key) as u32;
        let last = *self.buf.last().unwrap_or(&0) as u32;
        self.ctxh[NORD] = field
            ^ self.vpos.wrapping_mul(0x85eb_ca6b)
            ^ in_val_str.wrapping_mul(0xc2b2_ae35);
        self.ctxh[NORD + 1] = field.wrapping_mul(0x9e37_79b1)
            ^ depth.wrapping_mul(0x27d4_eb2f)
            ^ last.wrapping_mul(0x1656_67b1);
    }

    #[inline]
    fn predict(&mut self) -> f64 {
        for m in 0..NTAB {
            let idx = (self.ctxh[m] ^ self.c0.wrapping_mul(2_654_435_761)) as usize & MASK;
            self.idx[m] = idx;
            let p = self.t[m][idx] as f64 / 4096.0;
            self.st[m] = stretch(p);
        }
        // match model
        let mut pmatch = 0.5;
        if self.mm_on {
            let bp = self.bitpos;
            let placed = self.c0 - (1 << bp);
            if bp == 0 || placed == (self.pb as u32 >> (8 - bp)) {
                let predbit = (self.pb >> (7 - bp)) & 1;
                let ml = self.match_len as f64;
                let conf = (ml / (ml + 1.0)).min(0.97);
                pmatch = if predbit == 1 { 0.5 + 0.49 * conf } else { 0.5 - 0.49 * conf };
            }
        }
        self.st[NTAB] = stretch(pmatch);
        // numeric model
        let mut pnum = 0.5;
        if self.np_active && self.np_ptr < self.np_len {
            let pbn = self.np_digits[self.np_ptr];
            let bp = self.bitpos;
            let placed = self.c0 - (1 << bp);
            if bp == 0 || placed == (pbn as u32 >> (8 - bp)) {
                let predbit = (pbn >> (7 - bp)) & 1;
                pnum = if predbit == 1 { 0.5 + 0.49 * self.np_conf } else { 0.5 - 0.49 * self.np_conf };
            }
        }
        self.st[NTAB + 1] = stretch(pnum);

        self.wsel = *self.buf.last().unwrap_or(&0) as usize;
        let w = &self.w[self.wsel];
        let mut dot = 0.0;
        for i in 0..NIN {
            dot += w[i] * self.st[i];
        }
        self.p1 = squash(dot).clamp(1.0 / 4096.0, 4095.0 / 4096.0);
        self.p1
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let err = bit as f64 - self.p1;
        let w = &mut self.w[self.wsel];
        for i in 0..NIN {
            w[i] += LR * err * self.st[i];
        }
        let target = (bit * 4096) as i32;
        for m in 0..NTAB {
            let cell = &mut self.t[m][self.idx[m]];
            let cur = *cell as i32;
            *cell = (cur + ((target - cur) >> RATE)) as u16;
        }
        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        if self.c0 >= 256 {
            let byte = (self.c0 - 256) as u8;
            self.byte_boundary(byte);
            self.c0 = 1;
            self.bitpos = 0;
        }
    }

    fn byte_boundary(&mut self, byte: u8) {
        // --- match model ---
        let predicted_ok = self.mm_on && self.buf[self.match_ptr] == byte;
        self.buf.push(byte);
        let n = self.buf.len();
        if predicted_ok {
            self.match_ptr += 1;
            self.match_len += 1;
            if self.match_ptr >= n {
                self.mm_on = false;
                self.match_len = 0;
            }
        } else {
            self.mm_on = false;
            self.match_len = 0;
        }
        if n >= MINLEN {
            let hh = hash_n(&self.buf, n, MINLEN) as usize & MASK;
            if !self.mm_on {
                let cand = self.mm[hh] as usize;
                if cand != 0 && cand < n {
                    self.match_ptr = cand;
                    self.mm_on = true;
                    self.match_len = 0;
                }
            }
            self.mm[hh] = n as u32;
        }
        if self.mm_on && self.match_ptr < self.buf.len() {
            self.pb = self.buf[self.match_ptr];
        } else {
            self.mm_on = false;
        }

        // --- streaming JSON parser + numeric model ---
        self.update_struct(byte);

        self.recompute_ctx();
    }

    /// Set up the numeric prediction for the value about to be read in `field`.
    fn set_np(&mut self, field: u32) {
        let slot = self.num[(field as usize) & (NUMSLOTS - 1)];
        // Activate as soon as a field has been seen once; confidence (np_conf,
        // from the hit counter) scales how hard the mixer leans on it. Empirically
        // this beats a hard hit-threshold gate, which cost more on truly sequential
        // fields than it saved on noisy ones.
        if !slot.seen {
            self.np_active = false;
            return;
        }
        let pred = slot.last.wrapping_add(slot.delta);
        let neg = pred < 0;
        let mut x = (pred as i128).unsigned_abs();
        let mut d = [0u8; 24];
        let mut dl = 0;
        if x == 0 {
            d[0] = b'0';
            dl = 1;
        } else {
            while x > 0 {
                d[dl] = b'0' + (x % 10) as u8;
                x /= 10;
                dl += 1;
            }
        }
        let mut p = 0;
        if neg {
            self.np_digits[p] = b'-';
            p += 1;
        }
        for i in (0..dl).rev() {
            self.np_digits[p] = d[i];
            p += 1;
        }
        self.np_len = p;
        self.np_ptr = 0;
        self.np_active = true;
        let h = slot.hits as f64;
        self.np_conf = (h / (h + 1.0)).min(0.98);
    }

    fn finalize_numeric(&mut self) {
        if !self.cur_is_num || self.cur_len == 0 {
            return;
        }
        let actual = if self.cur_neg { -self.cur_num } else { self.cur_num };
        let i = (self.cur_field as usize) & (NUMSLOTS - 1);
        let slot = self.num[i];
        let predicted = slot.last.wrapping_add(slot.delta);
        let mut ns = slot;
        if slot.seen {
            ns.delta = actual.wrapping_sub(slot.last);
            ns.hits = if predicted == actual { (slot.hits + 1).min(255) } else { slot.hits / 2 };
        } else {
            ns.delta = 0;
            ns.hits = 0;
        }
        ns.last = actual;
        ns.seen = true;
        self.num[i] = ns;
    }

    #[inline]
    fn np_consume(&mut self, c: u8) {
        if self.np_active {
            if self.np_ptr < self.np_len && self.np_digits[self.np_ptr] == c {
                self.np_ptr += 1;
            } else {
                self.np_active = false;
            }
        }
    }

    fn update_struct(&mut self, c: u8) {
        // ---- inside a string ----
        if self.in_str {
            if self.esc {
                self.esc = false;
                self.cur_str_hash = hstep(self.cur_str_hash, c);
            } else if c == b'\\' {
                self.esc = true;
            } else if c == b'"' {
                self.in_str = false;
                if self.str_is_key {
                    if let Some(f) = self.stack.last_mut() {
                        f.key_hash = self.cur_str_hash;
                        f.expect_key = false;
                    }
                }
            } else {
                self.cur_str_hash = hstep(self.cur_str_hash, c);
            }
            self.vpos = (self.vpos + 1).min(31);
            return;
        }

        // ---- accumulating a bare numeric value ----
        if self.in_num_value {
            if c.is_ascii_digit() {
                if self.cur_len < 18 {
                    self.cur_num = self.cur_num * 10 + (c - b'0') as i64;
                    self.cur_len += 1;
                } else {
                    self.cur_is_num = false;
                }
                self.np_consume(c);
                self.vpos = (self.vpos + 1).min(31);
                return;
            } else {
                // value ends here. If it's actually a float / scientific form
                // (`.`, `e`, `E`), don't treat it as an integer sequence — that
                // would poison the field's delta state with a bogus integer part.
                if c == b'.' || c == b'e' || c == b'E' {
                    self.cur_is_num = false;
                }
                self.finalize_numeric(); // skips when !cur_is_num
                self.in_num_value = false;
                self.np_active = false;
            }
        }

        // ---- a value is expected: classify its first byte ----
        if self.value_pending {
            match c {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    self.vpos = 0;
                    return; // keep pending
                }
                b'0'..=b'9' | b'-' => {
                    self.value_pending = false;
                    self.in_num_value = true;
                    self.cur_is_num = true;
                    self.cur_neg = c == b'-';
                    self.cur_num = 0;
                    self.cur_len = 0;
                    self.cur_field = self.field_hash();
                    if c != b'-' {
                        self.cur_num = (c - b'0') as i64;
                        self.cur_len = 1;
                    }
                    self.np_consume(c);
                    self.vpos = 0;
                    return;
                }
                _ => {
                    self.value_pending = false;
                    self.np_active = false;
                    // fall through: structural / string / container
                }
            }
        }

        // ---- structural ----
        match c {
            b'"' => {
                self.in_str = true;
                self.esc = false;
                self.cur_str_hash = 0x9e37_79b1;
                self.str_is_key = self
                    .stack
                    .last()
                    .map_or(false, |f| f.is_object && f.expect_key);
                self.vpos = 0;
            }
            b'{' => {
                self.stack.push(Frame { is_object: true, key_hash: 0, expect_key: true });
                self.vpos = 0;
            }
            b'[' => {
                self.stack.push(Frame { is_object: false, key_hash: 0, expect_key: false });
                self.value_pending = true;
                let f = self.field_hash();
                self.set_np(f);
                self.vpos = 0;
            }
            b'}' | b']' => {
                self.stack.pop();
                self.vpos = 0;
            }
            b':' => {
                if let Some(f) = self.stack.last_mut() {
                    f.expect_key = false;
                }
                self.value_pending = true;
                let f = self.field_hash();
                self.set_np(f);
                self.vpos = 0;
            }
            b',' => {
                let in_obj = self.stack.last().map_or(false, |f| f.is_object);
                if in_obj {
                    if let Some(f) = self.stack.last_mut() {
                        f.expect_key = true;
                    }
                    self.value_pending = false;
                    self.np_active = false;
                } else {
                    self.value_pending = true;
                    let f = self.field_hash();
                    self.set_np(f);
                }
                self.vpos = 0;
            }
            b' ' | b'\n' | b'\r' | b'\t' => {
                self.vpos = 0;
            }
            _ => {
                self.vpos = (self.vpos + 1).min(31);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level compress / decompress
// ---------------------------------------------------------------------------

#[inline]
fn quantize(p1: f64) -> u32 {
    ((p1 * 4096.0).round() as i32).clamp(1, 4095) as u32
}

fn compress(data: &[u8]) -> Vec<u8> {
    let mut pr = Predictor::new();
    let mut enc = Encoder::new();
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p = quantize(pr.predict());
            enc.encode(bit, p);
            pr.update(bit);
        }
    }
    enc.finish()
}

fn decompress(comp: &[u8], orig_len: usize) -> Vec<u8> {
    let mut pr = Predictor::new();
    let mut dec = Decoder::new(comp);
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..orig_len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = quantize(pr.predict());
            let bit = dec.decode(p);
            pr.update(bit);
            byte = (byte << 1) | bit as u8;
        }
        out.push(byte);
    }
    out
}

fn main() {
    {
        let test = b"the quick brown fox the quick brown fox 1 2 3 4 5 6 7 8 9 10 11 12 13";
        let c = compress(test);
        let d = decompress(&c, test.len());
        assert!(d == test, "SELF-TEST ROUNDTRIP FAILED");
        eprintln!("self-test ok ({} -> {} bytes)", test.len(), c.len());
    }

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: augur <file> [sample_bytes]");
        return;
    }
    let path = &args[1];
    let limit: Option<usize> = args.get(2).and_then(|s| s.parse().ok());

    let mut data = fs::read(path).expect("read input");
    if let Some(l) = limit {
        data.truncate(l);
    }

    let t0 = Instant::now();
    let comp = compress(&data);
    let enc_t = t0.elapsed();

    let t1 = Instant::now();
    let dec = decompress(&comp, data.len());
    let dec_t = t1.elapsed();

    let ok = dec == data;
    let ratio = data.len() as f64 / comp.len() as f64;
    println!(
        "{}\n  {} -> {} bytes   ratio = {:.2}x   enc={:.1}s dec={:.1}s   roundtrip={}",
        path,
        data.len(),
        comp.len(),
        ratio,
        enc_t.as_secs_f64(),
        dec_t.as_secs_f64(),
        if ok { "OK" } else { "*** FAILED ***" }
    );
    if !ok {
        std::process::exit(1);
    }
}
