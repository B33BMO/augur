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
//!   - NUMERIC model: per-field linear extrapolation (predicts digits of
//!     last + delta before they are read). Formula detection for IDs/timestamps.
//!
//! Math is integer fixed-point: stretch/squash are lookup tables and the mixer
//! runs in i32/i64, so the inner loop has no transcendental calls. Probabilities
//! are 12-bit (0..4096); mixer weights are 16.16 fixed-point.

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
// stretch / squash lookup tables (12-bit prob <-> stretched logit domain).
// ---------------------------------------------------------------------------

const ST_MIN: i32 = -2047;
const ST_MAX: i32 = 2047;

fn build_stretch() -> Vec<i32> {
    // stretch(p) = 256 * ln(p / (4096 - p)), clamped to [-2047, 2047]
    (0..4096)
        .map(|p| {
            let pc = (p as f64).clamp(1.0, 4095.0);
            (256.0 * (pc / (4096.0 - pc)).ln()).round().clamp(ST_MIN as f64, ST_MAX as f64) as i32
        })
        .collect()
}

fn build_squash() -> Vec<i32> {
    // squash(d) = 4096 / (1 + e^(-d/256)); index i represents d = i - 2048
    (0..4096)
        .map(|i| {
            let d = (i - 2048) as f64;
            (4096.0 / (1.0 + (-d / 256.0).exp())).round().clamp(1.0, 4095.0) as i32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Predictor: portfolio of models + logistic mixer (integer fixed-point).
// ---------------------------------------------------------------------------

const MEM_BITS: usize = 20;
const MASK: usize = (1 << MEM_BITS) - 1;
const NORD: usize = 5;
const NSTR: usize = 2;
const NTAB: usize = NORD + NSTR;
const NIN: usize = NTAB + 2; // + match + numeric
const MINLEN: usize = 6;
const RATE: i32 = 4;
const LR: i32 = 7; // mixer learning rate (lpaq-style)
const ARRAY_TAG: u32 = 0xA22A_5151;
const NUMSLOTS: usize = 1 << 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generic,
    Json,
    Csv,
}

impl Mode {
    fn to_byte(self) -> u8 {
        match self {
            Mode::Generic => 0,
            Mode::Json => 1,
            Mode::Csv => 2,
        }
    }
    fn from_byte(b: u8) -> Mode {
        match b {
            1 => Mode::Json,
            2 => Mode::Csv,
            _ => Mode::Generic,
        }
    }
}

/// Sniff the data format from a prefix. The result is stored in the container
/// header so the decoder configures the same parser — the JSON and CSV models
/// never run at once and so never interfere.
fn sniff(data: &[u8]) -> Mode {
    let sample = &data[..data.len().min(65536)];
    let first = sample.iter().copied().find(|b| !b.is_ascii_whitespace());
    let mut lines = 1usize;
    let mut commas = 0usize;
    let mut braces = 0usize;
    for &b in sample {
        match b {
            b'\n' => lines += 1,
            b',' => commas += 1,
            b'{' => braces += 1,
            _ => {}
        }
    }
    if matches!(first, Some(b'{') | Some(b'[')) && braces * 2 >= lines {
        Mode::Json
    } else if commas >= lines {
        Mode::Csv
    } else {
        Mode::Generic
    }
}

/// Field identity for a CSV column (kept distinct from JSON field hashes).
#[inline]
fn csv_field(col: u32) -> u32 {
    col.wrapping_mul(0x9e37_79b1) ^ 0xC5C5_3737
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

/// Confidence magnitude in 12-bit prob space from a hit/run count: 2048 -> 4095.
#[inline]
fn conf_prob(count: u32) -> usize {
    (2048 + (2047 * count / (count + 1)) as i32).clamp(2048, 4095) as usize
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
    t: Vec<u16>, // NTAB context tables, flattened: model m occupies [m<<MEM_BITS ..]
    mm: Vec<u32>,
    stretch_tab: Vec<i32>,
    squash_tab: Vec<i32>,
    // bit-assembly state
    c0: u32,
    bitpos: u32,
    ctxh: [u32; NTAB],
    // match model
    mm_on: bool,
    match_ptr: usize,
    match_len: u32,
    pb: u8,
    match_mag: i32, // stretched confidence (>=0), recomputed at byte boundary
    // streaming JSON parser
    in_str: bool,
    esc: bool,
    str_is_key: bool,
    cur_str_hash: u32,
    stack: Vec<Frame>,
    vpos: u32,
    value_pending: bool,
    // format mode + CSV parser state
    mode: Mode,
    csv_col: u32,
    csv_in_quote: bool,
    csv_value_pending: bool,
    // numeric model
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
    num_mag: i32, // stretched confidence (>=0)
    // mixer (16.16 fixed-point weights, one set per previous byte)
    w: Vec<[i32; NIN]>,
    // cached for update()
    idx: [usize; NTAB],
    st: [i32; NIN],
    wsel: usize,
    pr: i32, // last predicted P(bit=1), 12-bit
}

impl Predictor {
    fn new(mode: Mode) -> Self {
        let init_w = (1i32 << 16) / NIN as i32;
        let mut p = Self {
            buf: Vec::new(),
            t: vec![2048u16; NTAB << MEM_BITS],
            mm: vec![0u32; 1 << MEM_BITS],
            stretch_tab: build_stretch(),
            squash_tab: build_squash(),
            c0: 1,
            bitpos: 0,
            ctxh: [0; NTAB],
            mm_on: false,
            match_ptr: 0,
            match_len: 0,
            pb: 0,
            match_mag: 0,
            in_str: false,
            esc: false,
            str_is_key: false,
            cur_str_hash: 0,
            stack: Vec::with_capacity(32),
            vpos: 0,
            value_pending: false,
            mode,
            csv_col: 0,
            csv_in_quote: false,
            csv_value_pending: true,
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
            num_mag: 0,
            w: vec![[init_w; NIN]; 256],
            idx: [0; NTAB],
            st: [0; NIN],
            wsel: 0,
            pr: 2048,
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
        // structure context: (field identity, secondary axis) per format
        let last = *self.buf.last().unwrap_or(&0) as u32;
        let (field, aux) = match self.mode {
            Mode::Json => (self.field_hash(), self.stack.len().min(15) as u32),
            Mode::Csv => (csv_field(self.csv_col), self.csv_col),
            Mode::Generic => (0, 0),
        };
        let in_val_str = (self.in_str && !self.str_is_key) as u32;
        self.ctxh[NORD] = field
            ^ self.vpos.wrapping_mul(0x85eb_ca6b)
            ^ in_val_str.wrapping_mul(0xc2b2_ae35);
        self.ctxh[NORD + 1] = field.wrapping_mul(0x9e37_79b1)
            ^ aux.wrapping_mul(0x27d4_eb2f)
            ^ last.wrapping_mul(0x1656_67b1);
    }

    #[inline]
    fn predict(&mut self) -> u32 {
        for m in 0..NTAB {
            let local = (self.ctxh[m] ^ self.c0.wrapping_mul(2_654_435_761)) as usize & MASK;
            let flat = (m << MEM_BITS) | local; // local < 2^MEM_BITS, so this is m*stride+local
            self.idx[m] = flat;
            // SAFETY: flat < NTAB<<MEM_BITS = t.len(); tv < 4096 = stretch_tab.len()
            let tv = unsafe { *self.t.get_unchecked(flat) } as usize;
            self.st[m] = unsafe { *self.stretch_tab.get_unchecked(tv & 4095) };
        }
        // match model: ± precomputed confidence depending on the predicted bit
        let mut sm = 0;
        if self.mm_on {
            let bp = self.bitpos;
            let placed = self.c0 - (1 << bp);
            if bp == 0 || placed == (self.pb as u32 >> (8 - bp)) {
                sm = if (self.pb >> (7 - bp)) & 1 == 1 { self.match_mag } else { -self.match_mag };
            }
        }
        self.st[NTAB] = sm;
        // numeric model
        let mut sn = 0;
        if self.np_active && self.np_ptr < self.np_len {
            let pbn = self.np_digits[self.np_ptr];
            let bp = self.bitpos;
            let placed = self.c0 - (1 << bp);
            if bp == 0 || placed == (pbn as u32 >> (8 - bp)) {
                sn = if (pbn >> (7 - bp)) & 1 == 1 { self.num_mag } else { -self.num_mag };
            }
        }
        self.st[NTAB + 1] = sn;

        // mix: dot product in 16.16 fixed-point
        self.wsel = *self.buf.last().unwrap_or(&0) as usize;
        // SAFETY: wsel < 256 = w.len(); (d+2048) in [1,4095] < squash_tab.len()
        let w = unsafe { self.w.get_unchecked(self.wsel) };
        let mut dot: i64 = 0;
        for i in 0..NIN {
            dot += w[i] as i64 * self.st[i] as i64;
        }
        let d = ((dot >> 16) as i32).clamp(ST_MIN, ST_MAX);
        self.pr = unsafe { *self.squash_tab.get_unchecked((d + 2048) as usize) }.clamp(1, 4095);
        self.pr as u32
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        // mixer weight update (integer gradient step)
        let err = (((bit as i32) << 12) - self.pr) * LR;
        // SAFETY: wsel < 256; each idx[m] = (m<<MEM_BITS)|local < t.len()
        let w = unsafe { self.w.get_unchecked_mut(self.wsel) };
        for i in 0..NIN {
            w[i] += (self.st[i] * err) >> 16;
        }
        // context table updates
        let target = (bit * 4096) as i32;
        for m in 0..NTAB {
            let cell = unsafe { self.t.get_unchecked_mut(self.idx[m]) };
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
            self.match_mag = self.stretch_tab[conf_prob(self.match_len)];
        } else {
            self.mm_on = false;
            self.match_mag = 0;
        }

        // --- structure + numeric model (format-aware) ---
        match self.mode {
            Mode::Json => self.update_struct_json(byte),
            Mode::Csv => self.update_struct_csv(byte),
            Mode::Generic => self.update_struct_generic(byte),
        }

        self.recompute_ctx();
    }

    fn set_np(&mut self, field: u32) {
        let slot = self.num[(field as usize) & (NUMSLOTS - 1)];
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
        self.num_mag = self.stretch_tab[conf_prob(slot.hits)];
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

    fn update_struct_json(&mut self, c: u8) {
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
                if c == b'.' || c == b'e' || c == b'E' {
                    self.cur_is_num = false; // float / scientific: don't track as int
                }
                self.finalize_numeric();
                self.in_num_value = false;
                self.np_active = false;
            }
        }

        if self.value_pending {
            match c {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    self.vpos = 0;
                    return;
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
                }
            }
        }

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

    /// Generic parser for unstructured/text data: tracks position-in-token and
    /// in-string state, which gives a cheap positional context (helps logs and
    /// free text). No field identity.
    fn update_struct_generic(&mut self, c: u8) {
        if self.in_str {
            if c == b'"' {
                self.in_str = false;
            }
            self.vpos = (self.vpos + 1).min(31);
            return;
        }
        match c {
            b'"' => {
                self.in_str = true;
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

    /// CSV/delimited parser: exposes the current column index as the semantic
    /// context, and routes the numeric model per-column (so sequential/integer
    /// columns get formula prediction just like JSON fields do).
    fn update_struct_csv(&mut self, c: u8) {
        if self.csv_in_quote {
            if c == b'"' {
                self.csv_in_quote = false;
            }
            self.in_num_value = false; // quoted field is not a bare number
            self.np_active = false;
            self.vpos = (self.vpos + 1).min(31);
            return;
        }
        match c {
            b'"' => {
                self.csv_in_quote = true;
                self.csv_value_pending = false;
                self.in_num_value = false;
                self.np_active = false;
                self.vpos = 0;
            }
            b',' | b'\n' => {
                if self.in_num_value {
                    self.finalize_numeric();
                    self.in_num_value = false;
                }
                self.np_active = false;
                if c == b'\n' {
                    self.csv_col = 0;
                } else {
                    self.csv_col += 1;
                }
                let f = csv_field(self.csv_col);
                self.set_np(f); // set up prediction for the next column's value
                self.csv_value_pending = true;
                self.vpos = 0;
            }
            _ => {
                if self.csv_value_pending {
                    self.csv_value_pending = false;
                    if c.is_ascii_digit() || c == b'-' {
                        self.in_num_value = true;
                        self.cur_is_num = true;
                        self.cur_neg = c == b'-';
                        self.cur_num = 0;
                        self.cur_len = 0;
                        self.cur_field = csv_field(self.csv_col);
                        if c != b'-' {
                            self.cur_num = (c - b'0') as i64;
                            self.cur_len = 1;
                        }
                        self.np_consume(c);
                    } else {
                        self.in_num_value = false;
                        self.np_active = false;
                    }
                } else if self.in_num_value {
                    if c.is_ascii_digit() {
                        if self.cur_len < 18 {
                            self.cur_num = self.cur_num * 10 + (c - b'0') as i64;
                            self.cur_len += 1;
                        } else {
                            self.cur_is_num = false;
                        }
                        self.np_consume(c);
                    } else {
                        if c == b'.' || c == b'e' || c == b'E' {
                            self.cur_is_num = false;
                        }
                        self.finalize_numeric();
                        self.in_num_value = false;
                        self.np_active = false;
                    }
                }
                self.vpos = (self.vpos + 1).min(31);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level compress / decompress
// ---------------------------------------------------------------------------

// Container layout: "AUGR" | version(1) | mode(1) | orig_len(8, LE) | stream
const MAGIC: [u8; 4] = *b"AUGR";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 14;

fn encode_stream(data: &[u8], mode: Mode) -> Vec<u8> {
    let mut pr = Predictor::new(mode);
    let mut enc = Encoder::new();
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p = pr.predict();
            enc.encode(bit, p);
            pr.update(bit);
        }
    }
    enc.finish()
}

fn decode_stream(stream: &[u8], mode: Mode, orig_len: usize) -> Vec<u8> {
    let mut pr = Predictor::new(mode);
    let mut dec = Decoder::new(stream);
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..orig_len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = pr.predict();
            let bit = dec.decode(p);
            pr.update(bit);
            byte = (byte << 1) | bit as u8;
        }
        out.push(byte);
    }
    out
}

fn compress(data: &[u8]) -> Vec<u8> {
    let mode = sniff(data);
    let stream = encode_stream(data, mode);
    let mut out = Vec::with_capacity(stream.len() + HEADER_LEN);
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(mode.to_byte());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&stream);
    out
}

fn decompress(container: &[u8]) -> Result<Vec<u8>, String> {
    if container.len() < HEADER_LEN || container[0..4] != MAGIC {
        return Err("not an augur file (bad magic)".into());
    }
    if container[4] != VERSION {
        return Err(format!("unsupported augur version {}", container[4]));
    }
    let mode = Mode::from_byte(container[5]);
    let orig_len = u64::from_le_bytes(container[6..14].try_into().unwrap()) as usize;
    Ok(decode_stream(&container[HEADER_LEN..], mode, orig_len))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("compress") | Some("c") => cmd_compress(&args[2..]),
        Some("decompress") | Some("d") => cmd_decompress(&args[2..]),
        Some("bench") => cmd_bench(&args[2..]),
        _ => {
            eprintln!("augur — structure-aware lossless compressor\n");
            eprintln!("usage:");
            eprintln!("  augur compress   <file> [-o out.augur]   compress to <file>.augur");
            eprintln!("  augur decompress <file.augur> [-o out]   restore the original");
            eprintln!("  augur bench      <file> [sample_bytes]   compress+verify+time in memory");
        }
    }
}

fn parse_io(args: &[String], default_out: impl Fn(&str) -> String) -> (String, String) {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            s if input.is_none() => input = Some(s.to_string()),
            _ => {}
        }
        i += 1;
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("error: no input file");
        std::process::exit(2);
    });
    let output = output.unwrap_or_else(|| default_out(&input));
    (input, output)
}

fn read_or_die(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {path}: {e}");
        std::process::exit(1);
    })
}

fn write_or_die(path: &str, data: &[u8]) {
    fs::write(path, data).unwrap_or_else(|e| {
        eprintln!("error: cannot write {path}: {e}");
        std::process::exit(1);
    });
}

fn cmd_compress(args: &[String]) {
    let (input, output) = parse_io(args, |i| format!("{i}.augur"));
    let data = read_or_die(&input);
    let t0 = Instant::now();
    let comp = compress(&data);
    let dt = t0.elapsed().as_secs_f64();
    write_or_die(&output, &comp);
    let ratio = if comp.is_empty() { 0.0 } else { data.len() as f64 / comp.len() as f64 };
    let mbps = data.len() as f64 / 1e6 / dt.max(1e-9);
    println!(
        "{input} ({} B) -> {output} ({} B)   ratio={ratio:.2}x   {mbps:.1} MB/s",
        data.len(),
        comp.len()
    );
}

fn cmd_decompress(args: &[String]) {
    let (input, output) = parse_io(args, |i| {
        i.strip_suffix(".augur").map(str::to_string).unwrap_or_else(|| format!("{i}.out"))
    });
    let comp = read_or_die(&input);
    let t0 = Instant::now();
    let data = decompress(&comp).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let dt = t0.elapsed().as_secs_f64();
    write_or_die(&output, &data);
    let mbps = data.len() as f64 / 1e6 / dt.max(1e-9);
    println!("{input} -> {output} ({} B)   {mbps:.1} MB/s", data.len());
}

fn cmd_bench(args: &[String]) {
    // self-test: prove roundtrip on a tiny mixed input first
    let test = b"the quick brown fox the quick brown fox 1 2 3 4 5 6 7 8 9 10 11 12 13";
    assert!(decompress(&compress(test)).unwrap() == test, "SELF-TEST ROUNDTRIP FAILED");

    let Some(input) = args.first().cloned() else {
        eprintln!("usage: augur bench <file> [sample_bytes]");
        return;
    };
    let limit: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
    let mut data = read_or_die(&input);
    if let Some(l) = limit {
        data.truncate(l);
    }

    let t0 = Instant::now();
    let comp = compress(&data);
    let enc_t = t0.elapsed();
    let t1 = Instant::now();
    let dec = decompress(&comp).unwrap();
    let dec_t = t1.elapsed();

    let ok = dec == data;
    let ratio = data.len() as f64 / comp.len() as f64;
    let enc_mbps = data.len() as f64 / 1e6 / enc_t.as_secs_f64();
    let dec_mbps = data.len() as f64 / 1e6 / dec_t.as_secs_f64();
    println!(
        "{input}\n  {} -> {} bytes   ratio={ratio:.2}x   enc={:.1}s ({enc_mbps:.1} MB/s) dec={:.1}s ({dec_mbps:.1} MB/s)   roundtrip={}",
        data.len(),
        comp.len(),
        enc_t.as_secs_f64(),
        dec_t.as_secs_f64(),
        if ok { "OK" } else { "*** FAILED ***" }
    );
    if !ok {
        std::process::exit(1);
    }
}
