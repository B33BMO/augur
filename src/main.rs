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
//!   - order 0,1,2,3,4,6,8 direct context models (local statistics)
//!   - WORD models: the token being typed, and that token after the previous one
//!   - MATCH models (hash chains): long-range repeats — what byte contexts can't
//!     see — with backward-context candidate selection
//!   - STRUCTURE models: a streaming parser exposes "which field's value am I
//!     inside" for JSON, CSV, SQL tuples, XML elements, and whitespace-delimited
//!     log columns; we condition on (field, position) and (field, depth).
//!   - RECORD-HISTORY model: replays the previous record's value for *this*
//!     field, from its first byte — the redundancy a byte-context model is
//!     structurally blind to, because the bytes just before a field's value
//!     belong to a different field.
//!   - NUMERIC model: per-field linear extrapolation (predicts digits of
//!     last + delta before they are read). Formula detection for IDs/timestamps,
//!     including cross-column relations within a row.
//!   - STRIDE/SPARSE models: a detected record period turns binary tables into
//!     columns; sparse contexts skip bytes to see interleaved fields.
//!
//! The oracles (match, record, numeric) do not assert a confidence — a TrustMap
//! learns, per situation, how often each has actually been right. A two-layer
//! mixer blends everything, and an SSE/APM chain calibrates the result before it
//! reaches the coder.
//!
//! Math is integer fixed-point: stretch/squash are lookup tables and the mixer
//! runs in i32/i64, so the inner loop has no transcendental calls. Probabilities
//! are 12-bit (0..4096); mixer weights are 16.16 fixed-point.

use std::env;
use std::fs;
use std::sync::OnceLock;
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

/// Both tables are pure functions of nothing, so they are built once per process
/// and shared. Keeping them out of `Predictor` also lets models borrow them while
/// the predictor holds a mutable borrow of itself.
fn stretch_tab() -> &'static [i32] {
    static T: OnceLock<Vec<i32>> = OnceLock::new();
    T.get_or_init(build_stretch)
}

fn squash_tab() -> &'static [i32] {
    static T: OnceLock<Vec<i32>> = OnceLock::new();
    T.get_or_init(build_squash)
}

/// squash() with the argument in stretched units rather than a table index.
#[inline]
fn squash(d: i32) -> i32 {
    squash_tab()[(d.clamp(ST_MIN, ST_MAX) + 2048) as usize]
}

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

/// Table size is chosen from the input length and recorded in the header, so a
/// small file doesn't pay for a large file's tables. Collisions in the context
/// tables cost real ratio, so the cap is generous — but there is no point
/// allocating 200 MB of slots for a 4 KB input that can never touch them.
const MEM_BITS_MIN: usize = 16;
const MEM_BITS_MAX: usize = 22;

fn mem_bits_for(len: usize) -> usize {
    let bl = usize::BITS - len.max(1).leading_zeros(); // ceil-ish log2
    (bl as usize + 2).clamp(MEM_BITS_MIN, MEM_BITS_MAX)
}
/// Byte-context orders modelled, in addition to order 0. Skipping 5 and 7 keeps
/// the portfolio cheap: adjacent orders are highly correlated, so the marginal
/// value of order 5 next to 4 and 6 is small compared to its table cost.
const ORDERS: [usize; 6] = [1, 2, 3, 4, 6, 8];
const NORD: usize = 1 + ORDERS.len();
const NWORD: usize = 2; // word model + (previous word, word) model
const NSTR: usize = 2;
const NBIN: usize = 4; // 2 record-stride + 2 sparse contexts
const NTAB: usize = NORD + NWORD + NSTR + NBIN;
const WORD0: usize = NORD; // first word-model slot in ctxh
const STR0: usize = NORD + NWORD; // first structure-model slot in ctxh
const BIN0: usize = STR0 + NSTR; // first binary/record-stride slot in ctxh
const NMATCH: usize = 2; // match models: short-context (fast reacquire) + long (locks long repeats)
const NREC: usize = 1; // record-history model
const NIN: usize = NTAB + NMATCH + NREC + 1; // tables + match + record + numeric
const MINLEN: usize = 6;
const MINLEN_LONG: usize = 16;
const CLIMIT: u16 = 12; // counter saturation: caps the slowest adaptation rate
const LR: i32 = 15; // mixer learning rate (retuned for the two-layer mixer)
const APM_RATE: i32 = 7; // SSE adaptation shift
const NMIX: usize = 4; // layer-1 mixers, one per selector view
/// log2(weight sets) for each layer-1 mixer: partial byte, previous byte,
/// match-state x bit position, structure context.
/// Mixer 2's selector packs a 4-bit match-length bucket, three oracle on-flags
/// and a 3-bit bit position — ten bits. Anything narrower silently masks the
/// length bucket away, which is the most informative thing there: how much to
/// trust the match model is almost entirely a question of how long the match is.
const MIX_CTX_BITS: [usize; NMIX] = [8, 8, 10, 8];
const ARRAY_TAG: u32 = 0xA22A_5151;
const NUMSLOTS: usize = 1 << 16;
const REC_MAXLEN: u32 = 8192; // longest value the record model will remember
const MAXCOL: usize = 32; // columns tracked per row for cross-column formula detection

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generic,
    Json,
    Csv,
    Sql,
    Xml,
}

impl Mode {
    fn to_byte(self) -> u8 {
        match self {
            Mode::Generic => 0,
            Mode::Json => 1,
            Mode::Csv => 2,
            Mode::Sql => 3,
            Mode::Xml => 4,
        }
    }
    fn from_byte(b: u8) -> Mode {
        match b {
            1 => Mode::Json,
            2 => Mode::Csv,
            3 => Mode::Sql,
            4 => Mode::Xml,
            _ => Mode::Generic,
        }
    }
}

#[inline]
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Does the sample look like delimited columns? Counting commas is not enough —
/// prose has commas too, and binary data has them by accident. What distinguishes
/// a table is that *every line has the same number of them*.
fn looks_like_csv(sample: &[u8]) -> bool {
    let mut counts: Vec<usize> = Vec::new();
    let mut cur = 0usize;
    for &b in sample {
        if b == b'\n' {
            counts.push(cur);
            cur = 0;
            if counts.len() >= 256 {
                break;
            }
        } else if b == b',' {
            cur += 1;
        }
    }
    if counts.len() < 4 {
        return false;
    }
    // the modal field count must be non-trivial and near-universal
    let mut best = 0usize;
    let mut best_n = 0usize;
    for &c in &counts {
        let n = counts.iter().filter(|&&x| x == c).count();
        if n > best_n {
            best_n = n;
            best = c;
        }
    }
    best >= 1 && best_n * 10 >= counts.len() * 8
}

/// Sniff the data format from a prefix. The result is stored in the container
/// header so the decoder configures the same parser — the format parsers never
/// run at once and so never interfere.
fn sniff(data: &[u8]) -> Mode {
    let sample = &data[..data.len().min(65536)];
    if sample.is_empty() {
        return Mode::Generic;
    }
    // A structured-text parser applied to binary produces nonsense field
    // identities, so gate every text mode behind an actual text check.
    let printable = sample
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        .count();
    if printable * 10 < sample.len() * 9 {
        return Mode::Generic;
    }
    let first = sample.iter().copied().find(|b| !b.is_ascii_whitespace());
    let lines = sample.iter().filter(|&&b| b == b'\n').count() + 1;
    let braces = sample.iter().filter(|&&b| b == b'{').count();

    if matches!(first, Some(b'{') | Some(b'[')) && braces * 2 >= lines {
        Mode::Json
    } else if contains(sample, b"</")
        && sample.iter().filter(|&&b| b == b'<').count() * 32 >= sample.len()
    {
        // Density rather than position: markup is often wrapped — Silesia's `xml`
        // is a tar, so its first byte is a filename, not a tag. Requiring closing
        // tags *and* roughly one tag per 32 bytes keeps source code out, where a
        // stray `<` is a comparison operator rather than an element.
        Mode::Xml
    } else if contains(sample, b"INSERT INTO") || contains(sample, b"CREATE TABLE") {
        Mode::Sql
    } else if looks_like_csv(sample) {
        Mode::Csv
    } else {
        Mode::Generic
    }
}

/// Field identity for a whitespace-delimited token position in generic text.
/// This is what turns an unlabelled log line into a table: field 4 of every
/// nginx line is the status code, whether or not anything says so.
#[inline]
fn gen_field(tok: u32) -> u32 {
    tok.wrapping_mul(0x9e37_79b1) ^ 0x1234_abcd
}

/// Field identity for a CSV column (kept distinct from JSON field hashes).
#[inline]
fn csv_field(col: u32) -> u32 {
    col.wrapping_mul(0x9e37_79b1) ^ 0xC5C5_3737
}

/// Field identity for a SQL tuple column at a given paren depth.
#[inline]
fn sql_field(col: u32, depth: u32) -> u32 {
    col.wrapping_mul(0x9e37_79b1) ^ depth.wrapping_mul(0x85eb_ca6b) ^ 0x5917_9179
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

// ---------------------------------------------------------------------------
// Adaptive counters.
//
// A context slot is a u32: the high 22 bits are P(bit==1) in 22-bit fixed point,
// the low 10 bits are a hit count. The update step moves the probability by
// `error / (n + 1.5)` — so a slot seen for the first time jumps almost all the
// way to the observed bit, while a slot with a long history barely moves. A
// fixed shift (the previous `>> 4`) had to compromise between those two, and
// paid for it on every freshly-touched context — which, with seven models over a
// megabyte of slots each, is most of them.
// ---------------------------------------------------------------------------

const CTR_INIT: u16 = 2048 << 4; // p = 1/2, unseen

/// RATE_TAB[n] = 65536 / (n + 1.5), i.e. the 16.16 fixed-point step fraction.
static RATE_TAB: [i32; 1024] = build_rate_tab();

const fn build_rate_tab() -> [i32; 1024] {
    let mut t = [0i32; 1024];
    let mut i = 0;
    while i < 1024 {
        t[i] = 131_072 / (2 * i as i32 + 3); // 65536 / (i + 1.5)
        i += 1;
    }
    t
}

/// The 12-bit probability held in a counter slot.
#[inline]
fn ctr_p12(v: u16) -> usize {
    (v >> 4) as usize
}

/// Move a counter slot toward `bit` at the rate its hit count earns.
#[inline]
fn ctr_update(v: u16, bit: u32, limit: u16) -> u16 {
    let n = v & 15;
    let p = (v >> 4) as i32;
    // SAFETY: n is masked to 0..15, well inside RATE_TAB.len()
    let rate = unsafe { *RATE_TAB.get_unchecked(n as usize) };
    let p = (p + (((((bit as i32) << 12) - p) * rate) >> 16)).clamp(0, 4095) as u16;
    let n = if n < limit { n + 1 } else { n };
    (p << 4) | n
}

// ---------------------------------------------------------------------------
// TrustMap — learned reliability for a byte-level oracle.
//
// The match, record and numeric models are *oracles*: each names the byte it
// thinks comes next. The question is how much to believe it, and the honest
// answer varies — a match model 200 bytes into a repeat is near-certain, one
// that just reacquired is a coin flip, and a record model replaying a timestamp
// is reliable for fifteen characters and worthless after that.
//
// augur used to answer with a hand-tuned confidence curve. A TrustMap answers by
// measuring: it keeps a counter per (agreement length, bit position, predicted
// bit) and learns the empirical probability that the oracle is right there. Each
// oracle then calibrates itself to the data at hand.
// ---------------------------------------------------------------------------

// A TrustMap holds only a few hundred counters, so unlike the context tables it
// can afford a wide slot: 22-bit probability and a 10-bit count. The long count
// is the point — how reliable a 200-byte match is does not drift, and a 4-bit
// count would force it to keep relearning that.
const TRUST_LIMIT: u32 = 255;
const TRUST_INIT: u32 = (1 << 21) << 10;

#[inline]
fn wide_p12(v: u32) -> usize {
    (v >> 20) as usize
}

#[inline]
fn wide_update(v: u32, bit: u32) -> u32 {
    let n = v & 1023;
    let p22 = (v >> 10) as i32;
    // SAFETY: n is masked to 0..1023 = RATE_TAB.len()
    let rate = unsafe { *RATE_TAB.get_unchecked(n as usize) };
    let err = (((bit as i32) << 22) - p22) as i64;
    let p22 = (p22 + ((err * rate as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
    let n = if n < TRUST_LIMIT { n + 1 } else { n };
    (p22 << 10) | n
}

struct TrustMap {
    t: Vec<u32>,
    idx: usize,
    active: bool,
    stab: &'static [i32],
}

impl TrustMap {
    fn new() -> Self {
        Self { t: vec![TRUST_INIT; TRUST_CTX], idx: 0, active: false, stab: stretch_tab() }
    }

    /// Stretched P(bit==1) given the oracle's state; remembers the cell to train.
    #[inline]
    fn predict(&mut self, cx: usize) -> i32 {
        self.idx = cx & (TRUST_CTX - 1);
        self.active = true;
        // SAFETY: idx is masked into range; ctr_p12 < 4096 = stretch_tab.len()
        unsafe { *self.stab.get_unchecked(wide_p12(*self.t.get_unchecked(self.idx))) }
    }

    /// The oracle had nothing to say this bit, so there is nothing to learn.
    #[inline]
    fn idle(&mut self) {
        self.active = false;
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        if self.active {
            let cell = unsafe { self.t.get_unchecked_mut(self.idx) };
            *cell = wide_update(*cell, bit);
        }
    }
}

/// (length bucket 0..15) x (bit position 0..7) x (predicted bit 0..1), with two
/// spare high bits for an oracle-specific axis (the record model uses them for
/// the field's reliability).
const TRUST_CTX: usize = 16 * 8 * 2 * 4;

#[inline]
fn trust_cx(lenbucket: u32, bitpos: u32, expected_bit: u32) -> usize {
    ((lenbucket.min(15) << 4) | (bitpos << 1) | expected_bit) as usize
}

/// Bucket a run length logarithmically: 0, 1, 2, 3-4, 5-8, ...
#[inline]
fn lenbucket(n: u32) -> u32 {
    (32 - n.leading_zeros()).min(15)
}

// ---------------------------------------------------------------------------
// Record-stride detection.
//
// Binary tables — a star catalogue, a database page, a raster scanline, a struct
// array — repeat with a fixed period that nothing in the file declares. Byte
// orders read along the stream and see noise; the useful neighbour of a value is
// the one one *record* above it, not one byte behind it.
//
// So watch how far apart four-byte patterns recur, and let the winning distance
// vote itself into being the record length. Text files simply never produce a
// sharp peak, and the models keyed on it stay silent.
// ---------------------------------------------------------------------------

const STRIDE_MAX: usize = 1024; // longest record period considered
const POS4_BITS: usize = 20;

struct StrideDetect {
    pos4: Vec<u32>, // hash of the last 4 bytes -> where they last appeared
    hist: Vec<u16>, // votes per candidate period
    stride: usize,
    best: u16,
    seen: u32,
}

impl StrideDetect {
    fn new() -> Self {
        Self {
            pos4: vec![0; 1 << POS4_BITS],
            hist: vec![0; STRIDE_MAX],
            stride: 0,
            best: 0,
            seen: 0,
        }
    }

    fn update(&mut self, buf: &[u8]) {
        let n = buf.len();
        if n < 4 {
            return;
        }
        let h = (hash_n(buf, n, 4) as usize) & ((1 << POS4_BITS) - 1);
        let last = self.pos4[h] as usize;
        self.pos4[h] = n as u32;
        if last == 0 {
            return;
        }
        let d = n - last;
        if !(2..STRIDE_MAX).contains(&d) {
            return;
        }
        self.hist[d] = self.hist[d].saturating_add(1);
        if self.hist[d] > self.best {
            self.best = self.hist[d];
            self.stride = d;
        }
        // decay, so a file whose layout changes partway can re-vote
        self.seen += 1;
        if self.seen >= 1 << 16 {
            self.seen = 0;
            self.best = 0;
            for v in self.hist.iter_mut() {
                *v /= 2;
            }
            self.best = self.hist[self.stride];
        }
    }
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

const MAX_CHAIN: usize = 8; // hash-chain candidates examined per acquire
const MAX_BACK: usize = 64; // backward context bytes compared to rank candidates

/// How far back the bytes before `cand` match the bytes before `cur` (capped).
#[inline]
fn backmatch(buf: &[u8], cand: usize, cur: usize, cap: usize) -> u32 {
    let mut k = 0;
    while k < cap && k < cand && k < cur && buf[cand - 1 - k] == buf[cur - 1 - k] {
        k += 1;
    }
    k as u32
}

/// A match-model predictor with hash chains: on a miss it walks the chain of
/// recent positions sharing the current `minlen`-byte context and picks the
/// candidate whose *preceding* bytes match the current context the longest — so
/// it locks onto genuine long repeats instead of the most-recent coincidence
/// (what let LZMA beat the single-position version on highly repetitive data).
struct MatchModel {
    head: Vec<u32>, // context hash -> latest position of the following byte (0 = none)
    prev: Vec<u32>, // (pos & mask) -> previous position in the chain (0 = end)
    mask: usize,
    minlen: usize,
    on: bool,
    ptr: usize,
    len: u32,
    pb: u8,
}

impl MatchModel {
    fn new(minlen: usize, mem_bits: usize) -> Self {
        Self {
            head: vec![0u32; 1 << mem_bits],
            prev: vec![0u32; 1 << mem_bits],
            mask: (1 << mem_bits) - 1,
            minlen,
            on: false,
            ptr: 0,
            len: 0,
            pb: 0,
        }
    }

    /// The bit this model expects next, or None when it is off or the bits
    /// already coded in this byte have diverged from its prediction.
    #[inline]
    fn expected(&self, c0: u32, bitpos: u32) -> Option<u32> {
        if !self.on {
            return None;
        }
        let placed = c0 - (1 << bitpos);
        if bitpos == 0 || placed == (self.pb as u32 >> (8 - bitpos)) {
            Some(((self.pb >> (7 - bitpos)) & 1) as u32)
        } else {
            None
        }
    }

    /// Advance at a byte boundary. `buf` already includes `byte`.
    fn update(&mut self, buf: &[u8], byte: u8) {
        let n = buf.len();
        // continue an active match
        if self.on && buf[self.ptr] == byte {
            self.ptr += 1;
            self.len += 1;
            if self.ptr >= n {
                self.on = false;
                self.len = 0;
            }
        } else {
            self.on = false;
            self.len = 0;
        }
        if n >= self.minlen {
            let h = hash_n(buf, n, self.minlen) as usize & self.mask;
            if !self.on {
                // walk the chain; pick the candidate with the longest backward context match
                let mut cand = self.head[h] as usize;
                let mut depth = 0;
                let mut best_pos = 0usize;
                let mut best_back = 0u32;
                while cand != 0 && cand < n && depth < MAX_CHAIN {
                    let back = backmatch(buf, cand, n, MAX_BACK);
                    if back > best_back {
                        best_back = back;
                        best_pos = cand;
                    }
                    let np = self.prev[cand & self.mask] as usize;
                    if np == 0 || np >= cand {
                        break; // end of chain / stale alias guard
                    }
                    cand = np;
                    depth += 1;
                }
                if best_back >= 1 {
                    self.ptr = best_pos;
                    self.on = true;
                    // chains pick a better *candidate*, but a fresh match starts only
                    // mildly confident so it can't override the structure/numeric models
                    // on structured data; a long ride still climbs to full confidence.
                    self.len = best_back.min(8);
                }
            }
            // link the current position into the chain for this context
            self.prev[n & self.mask] = self.head[h];
            self.head[h] = n as u32;
        }
        if self.on && self.ptr < n {
            self.pb = buf[self.ptr];
        } else {
            self.on = false;
        }
    }
}

/// Ask the CPU to start fetching a line we are about to need.
///
/// The inner loop's cost is dominated by fifteen scattered loads per bit into a
/// table far larger than cache. Those addresses are known one bit early — the
/// next partial byte can only be `c0<<1` or `c0<<1|1` — so both can be requested
/// before the mixer, the SSE chain and the coder run, which is enough work to
/// cover most of the latency. Purely a scheduling hint: it changes no output.
#[inline(always)]
fn prefetch(p: *const u16) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = p;
    }
}

// ---------------------------------------------------------------------------
// SSE / APM — secondary symbol estimation.
//
// The mixer produces a probability from a weighted blend, but that blend is
// systematically miscalibrated in ways that depend on context: "when I'm two
// bits into a byte inside a `created_at` value and I said 90%, I was actually
// right 97% of the time". An APM learns exactly that correction. It quantises
// the incoming probability into 33 buckets along the *stretched* axis (so the
// resolution is concentrated near 0 and 1, where it matters), looks up a learned
// output per (context, bucket), and interpolates between adjacent buckets.
// ---------------------------------------------------------------------------

struct Apm {
    t: Vec<u16>, // (context, bucket) -> refined probability, 16-bit
    idx: usize,  // cell touched by the last refine(), updated on the next bit
    mask: usize,
    stab: &'static [i32],
}

impl Apm {
    /// `bits` contexts, each with 33 interpolation buckets, initialised to the
    /// identity map so an untrained APM passes its input through unchanged.
    fn new(bits: usize) -> Self {
        let n = 1usize << bits;
        let sq = squash_tab();
        let mut t = vec![0u16; n * 33];
        for j in 0..33 {
            let v = (sq[(((j as i32 - 16) * 128 + 2048) as usize).min(4095)] * 16) as u16;
            for i in 0..n {
                t[i * 33 + j] = v;
            }
        }
        Self { t, idx: 0, mask: n - 1, stab: stretch_tab() }
    }

    #[inline]
    fn refine(&mut self, pr: i32, cx: usize) -> i32 {
        // SAFETY: pr is clamped to 1..4095 by every caller; s lands in 1..4095 so
        // (s>>7) is 0..31 and i+1 stays inside the context's 33-bucket row.
        let s = unsafe { *self.stab.get_unchecked(pr as usize) } + 2048;
        let w = s & 127;
        let i = (s >> 7) as usize + (cx & self.mask) * 33;
        self.idx = i + (w >> 6) as usize;
        let lo = unsafe { *self.t.get_unchecked(i) } as i32;
        let hi = unsafe { *self.t.get_unchecked(i + 1) } as i32;
        ((lo * (128 - w) + hi * w) >> 11).clamp(1, 4095)
    }

    #[inline]
    fn update(&mut self, bit: u32, rate: i32) {
        // nudge toward the observed bit, but stop just short of 0/65536 so a
        // single surprise can never cost an unbounded number of bits
        let g = ((bit as i32) << 16) + ((bit as i32) << rate) - (bit as i32) * 2;
        let cell = unsafe { self.t.get_unchecked_mut(self.idx) };
        *cell = (*cell as i32 + ((g - *cell as i32) >> rate)) as u16;
    }
}

// ---------------------------------------------------------------------------
// Logistic mixer.
//
// A mixer holds one weight vector per *context* and blends its inputs in the
// stretched domain, training the active vector by gradient descent on coding
// loss. Which context selects the weights matters enormously: the right weight
// for the match model is different at bit 0 of a byte than at bit 7, and
// different again inside a numeric field than inside prose.
//
// Rather than pick one selector, augur runs several mixers in parallel — each
// keyed on a different view of the state — and blends *their* outputs with a
// second-layer mixer. Each layer-1 mixer is trained on its own error, so a
// selector that is useless on this data simply stops being trusted by layer 2.
// ---------------------------------------------------------------------------

struct Mixer<const N: usize> {
    w: Vec<i32>, // nsets * N weights, 16.16 fixed point
    mask: usize,
    sel: usize, // offset of the weight set used by the last mix()
    sqtab: &'static [i32],
}

impl<const N: usize> Mixer<N> {
    fn new(bits: usize) -> Self {
        let nsets = 1usize << bits;
        Self { w: vec![(1i32 << 16) / N as i32; nsets * N], mask: nsets - 1, sel: 0, sqtab: squash_tab() }
    }

    /// Blend `st` under weight set `ctx`; returns the stretched prediction.
    #[inline]
    fn mix(&mut self, st: &[i32; N], ctx: usize) -> i32 {
        self.sel = (ctx & self.mask) * N;
        // SAFETY: sel + N <= (mask+1)*N = w.len()
        let w = unsafe { self.w.get_unchecked(self.sel..self.sel + N) };
        let mut dot: i64 = 0;
        for i in 0..N {
            dot += w[i] as i64 * st[i] as i64;
        }
        ((dot >> 16) as i32).clamp(ST_MIN, ST_MAX)
    }

    /// Gradient step on the weight set that produced `out`.
    #[inline]
    fn update(&mut self, st: &[i32; N], out: i32, bit: u32) {
        let sq = unsafe { *self.sqtab.get_unchecked((out.clamp(ST_MIN, ST_MAX) + 2048) as usize) };
        let err = (((bit as i32) << 12) - sq) * LR;
        // SAFETY: sel was set by the matching mix() call
        let w = unsafe { self.w.get_unchecked_mut(self.sel..self.sel + N) };
        for i in 0..N {
            w[i] += (st[i] * err) >> 16;
        }
    }
}

struct Predictor {
    buf: Vec<u8>,
    t: Vec<u16>, // NTAB context tables, flattened: model m occupies [m<<mem_bits ..]
    mem_bits: usize,
    mask: usize,
    // SSE stages, applied to the mixer output in sequence
    apm_c0: Apm,
    apm_o1: Apm,
    apm_str: Apm,
    apm_ora: Apm,
    // bit-assembly state
    c0: u32,
    bitpos: u32,
    ctxh: [u32; NTAB],
    // match models (short + long context)
    matches: Vec<MatchModel>,
    // learned reliability for each byte-level oracle
    trust_match: Vec<TrustMap>,
    trust_rec: TrustMap,
    trust_num: TrustMap,
    // record-period detector feeding the stride contexts
    stride: StrideDetect,
    // word model state (format-independent: words matter in prose, JSON keys and logs alike)
    word_hash: u32,
    prev_word: u32,
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
    // SQL parser state
    sql_col: u32,
    sql_depth: u32,
    sql_value_pending: bool,
    sql_col_stack: [u32; 33],
    // XML parser state
    xml_stack: Vec<u32>,
    xml_in_tag: bool,
    xml_in_attr: bool,
    xml_aq: u8,
    xml_cur_hash: u32,
    xml_reading: bool,
    xml_name_started: bool,
    xml_close: bool,
    xml_selfclose: bool,
    // record-history model: per field, where its value in the previous record lives
    rec_start: Vec<u32>,
    rec_len: Vec<u32>,
    rec_hits: Vec<u8>,
    rec_field: u32,     // field slot of the value currently being read
    rec_open: u32,      // buffer offset where that value began
    rec_have_open: bool,
    rec_ptr: usize, // read cursor into the previous record's value
    rec_end: usize,
    rec_on: bool,
    rec_matched: u32,
    rec_pb: u8,
    gen_tok: u32,  // whitespace-delimited token index, generic mode
    gen_sep: bool, // inside a run of whitespace (so runs count as one separator)
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
    num_hits: u32, // how reliable this field's extrapolation has been
    cur_col: usize, // column index of the value being read (usize::MAX = none/JSON)
    // cross-column formula state (CSV/SQL): per-row values + per-column hypothesis
    row_vals: [i64; MAXCOL],
    row_has: [bool; MAXCOL],
    xsrc: [u8; MAXCOL],  // source column this column is predicted from
    xoff: [i64; MAXCOL], // offset: value = row_vals[src] + xoff
    xhits: [u32; MAXCOL],
    // two-layer mixer: NMIX context-selected mixers, blended by a second layer
    mix1: Vec<Mixer<NIN>>,
    mix2: Mixer<NMIX>,
    m1out: [i32; NMIX],
    // cached for update()
    idx: [usize; NTAB],
    st: [i32; NIN],
    mix2out: i32, // layer-2 stretched output — what the SSE chain refines
}

impl Predictor {
    fn new(mode: Mode, mem_bits: usize) -> Self {
        let mut p = Self {
            buf: Vec::new(),
            t: vec![CTR_INIT; NTAB << mem_bits],
            mem_bits,
            mask: (1 << mem_bits) - 1,
            apm_c0: Apm::new(8),
            apm_o1: Apm::new(16),
            apm_str: Apm::new(16),
            apm_ora: Apm::new(14),
            c0: 1,
            bitpos: 0,
            ctxh: [0; NTAB],
            matches: vec![MatchModel::new(MINLEN, mem_bits), MatchModel::new(MINLEN_LONG, mem_bits)],
            trust_match: (0..NMATCH).map(|_| TrustMap::new()).collect(),
            trust_rec: TrustMap::new(),
            trust_num: TrustMap::new(),
            stride: StrideDetect::new(),
            word_hash: 0,
            prev_word: 0,
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
            sql_col: 0,
            sql_depth: 0,
            sql_value_pending: false,
            sql_col_stack: [0; 33],
            xml_stack: Vec::with_capacity(32),
            xml_in_tag: false,
            xml_in_attr: false,
            xml_aq: 0,
            xml_cur_hash: 0,
            xml_reading: false,
            xml_name_started: false,
            xml_close: false,
            xml_selfclose: false,
            rec_start: vec![0; NUMSLOTS],
            rec_len: vec![0; NUMSLOTS],
            rec_hits: vec![0; NUMSLOTS],
            rec_field: 0,
            rec_open: 0,
            rec_have_open: false,
            rec_ptr: 0,
            rec_end: 0,
            rec_on: false,
            rec_matched: 0,
            rec_pb: 0,
            gen_tok: 0,
            gen_sep: true,
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
            num_hits: 0,
            cur_col: usize::MAX,
            row_vals: [0; MAXCOL],
            row_has: [false; MAXCOL],
            xsrc: [0; MAXCOL],
            xoff: [0; MAXCOL],
            xhits: [0; MAXCOL],
            mix1: MIX_CTX_BITS.iter().map(|&b| Mixer::new(b)).collect(),
            mix2: Mixer::new(8),
            m1out: [0; NMIX],
            idx: [0; NTAB],
            st: [0; NIN],
            mix2out: 0,
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
        for (i, &k) in ORDERS.iter().enumerate() {
            self.ctxh[1 + i] = if n >= k {
                hash_n(&self.buf, n, k)
            } else {
                (k as u32).wrapping_mul(0x9e37_79b1)
            };
        }
        // word contexts: the token being typed, and that token in the company of
        // the one before it. Byte orders see "tio"; this sees "informatio" and
        // "the informatio", which is what actually pins down the next letter.
        self.ctxh[WORD0] = self.word_hash.wrapping_mul(0xa24b_af05);
        self.ctxh[WORD0 + 1] =
            self.word_hash ^ self.prev_word.wrapping_mul(0x7feb_352d);
        // record-stride contexts: the byte one record above, and the pair one and
        // two records above — the vertical equivalent of order-1 and order-2.
        // Sparse contexts skip a byte, which catches interleaved fields (a
        // 4-byte struct field read one byte at a time) that dense orders blur.
        let s = self.stride.stride;
        let b = |k: usize| if n > k { self.buf[n - k] as u32 } else { 0 };
        if s >= 2 && n > 2 * s {
            self.ctxh[BIN0] = b(s)
                .wrapping_mul(0x9e37_79b1)
                ^ ((n % s) as u32).wrapping_mul(0x85eb_ca6b);
            self.ctxh[BIN0 + 1] =
                b(s).wrapping_mul(0xc2b2_ae35) ^ b(2 * s).wrapping_mul(0x27d4_eb2f);
        } else {
            self.ctxh[BIN0] = 0x5a5a_1111;
            self.ctxh[BIN0 + 1] = 0x5a5a_2222;
        }
        self.ctxh[BIN0 + 2] = b(1).wrapping_mul(0x2545_f491) ^ b(3).wrapping_mul(0x9e37_79b1);
        self.ctxh[BIN0 + 3] = b(2).wrapping_mul(0x7feb_352d) ^ b(4).wrapping_mul(0x846c_a68b);
        // structure context: (field identity, secondary axis) per format
        let last = *self.buf.last().unwrap_or(&0) as u32;
        let (field, aux) = match self.mode {
            Mode::Json => (self.field_hash(), self.stack.len().min(15) as u32),
            Mode::Csv => (csv_field(self.csv_col), self.csv_col),
            Mode::Sql => (sql_field(self.sql_col, self.sql_depth), self.sql_col),
            Mode::Xml => {
                let tag = *self.xml_stack.last().unwrap_or(&0);
                let state = if self.xml_in_attr { 2u32 } else if self.xml_in_tag { 1 } else { 0 };
                (tag ^ state.wrapping_mul(0x68e3_1da4), self.xml_stack.len().min(15) as u32)
            }
            Mode::Generic => (0, 0),
        };
        let in_val_str = (self.in_str && !self.str_is_key) as u32;
        self.ctxh[STR0] = field
            ^ self.vpos.wrapping_mul(0x85eb_ca6b)
            ^ in_val_str.wrapping_mul(0xc2b2_ae35);
        self.ctxh[STR0 + 1] = field.wrapping_mul(0x9e37_79b1)
            ^ aux.wrapping_mul(0x27d4_eb2f)
            ^ last.wrapping_mul(0x1656_67b1);
    }

    #[inline]
    fn predict(&mut self) -> u32 {
        let stab = stretch_tab();
        for m in 0..NTAB {
            let local = (self.ctxh[m] ^ self.c0.wrapping_mul(2_654_435_761)) as usize & self.mask;
            let flat = (m << self.mem_bits) | local; // local < 2^mem_bits, so this is m*stride+local
            self.idx[m] = flat;
            // SAFETY: flat < NTAB<<mem_bits = t.len(); ctr_p12 < 4096 = stretch_tab.len()
            let tv = unsafe { *self.t.get_unchecked(flat) };
            self.st[m] = unsafe { *stab.get_unchecked(ctr_p12(tv)) };
        }
        // Start both candidate loads for the next bit. Skipped on the last bit of
        // a byte, where the byte boundary rewrites every context hash and the
        // addresses are not yet knowable.
        if self.c0 < 128 {
            let base = self.t.as_ptr();
            let (n0, n1) = (self.c0 << 1, (self.c0 << 1) | 1);
            for m in 0..NTAB {
                let stride = m << self.mem_bits;
                let h0 = (self.ctxh[m] ^ n0.wrapping_mul(2_654_435_761)) as usize & self.mask;
                let h1 = (self.ctxh[m] ^ n1.wrapping_mul(2_654_435_761)) as usize & self.mask;
                // SAFETY: both offsets are < NTAB<<mem_bits = t.len(); prefetch of
                // an in-bounds address has no architectural effect regardless.
                unsafe {
                    prefetch(base.add(stride | h0));
                    prefetch(base.add(stride | h1));
                }
            }
        }
        // Oracles: each names an expected bit; a TrustMap turns that into a
        // probability whose confidence was learned from how often this oracle has
        // been right in this situation, rather than assumed.
        for i in 0..NMATCH {
            self.st[NTAB + i] = match self.matches[i].expected(self.c0, self.bitpos) {
                Some(eb) => {
                    let cx = trust_cx(lenbucket(self.matches[i].len), self.bitpos, eb);
                    self.trust_match[i].predict(cx)
                }
                None => {
                    self.trust_match[i].idle();
                    0
                }
            };
        }
        // record-history model: bucket by how much of this value has replayed,
        // offset by the field's historical reliability
        self.st[NTAB + NMATCH] = match self.rec_expected(self.c0, self.bitpos) {
            Some(eb) => {
                // "how far in am I" and "is this field usually repetitive" are
                // different questions; summing them would blur both
                let hits = self.rec_hits[Self::rec_slot(self.rec_field)] as u32;
                let cx = trust_cx(lenbucket(self.rec_matched), self.bitpos, eb)
                    | (lenbucket(hits).min(3) as usize) << 8;
                self.trust_rec.predict(cx)
            }
            None => {
                self.trust_rec.idle();
                0
            }
        };
        // numeric model
        self.st[NTAB + NMATCH + NREC] = match self.np_expected() {
            Some(eb) => {
                let cx = trust_cx(lenbucket(self.num_hits), self.bitpos, eb);
                self.trust_num.predict(cx)
            }
            None => {
                self.trust_num.idle();
                0
            }
        };

        // --- two-layer mixing ---
        let last = *self.buf.last().unwrap_or(&0) as u32;
        let prev2 = if self.buf.len() >= 2 { self.buf[self.buf.len() - 2] as u32 } else { 0 };
        let sel = self.mixer_selectors(last);
        for j in 0..NMIX {
            self.m1out[j] = self.mix1[j].mix(&self.st, sel[j]);
        }
        self.mix2out = self.mix2.mix(&self.m1out, self.c0 as usize);

        // SSE chain: three calibrations of the mixed output, each blended 3:1
        // with its input so a cold APM can only nudge, never hijack.
        let o1cx = (self.c0 ^ last.wrapping_mul(0x2545_f491) ^ prev2.wrapping_mul(0x9e37_79b1)) as usize;
        let strcx = (self.c0.wrapping_mul(0x9e37_79b1) ^ self.ctxh[STR0]) as usize;

        // The oracle state is a strong calibration context in its own right:
        // "the match model is 40 bytes into a repeat and the record model agrees"
        // is a different confidence regime from "nothing is locked on", and the
        // mixer alone cannot express that as a probability correction.
        let m0 = &self.matches[0];
        let onflags = (m0.on as usize) | ((self.matches[1].on as usize) << 1) | ((self.rec_on as usize) << 2);
        let oracx = ((lenbucket(m0.len) as usize) << 10)
            | (onflags << 7)
            | ((self.bitpos as usize) << 4)
            | (self.c0 as usize & 15);

        let p = squash(self.mix2out).clamp(1, 4095);
        let p = (self.apm_c0.refine(p, self.c0 as usize) * 3 + p) >> 2;
        let p = (self.apm_o1.refine(p, o1cx >> 8) * 3 + p) >> 2;
        let p = (self.apm_str.refine(p, strcx >> 8) * 3 + p) >> 2;
        let p = (self.apm_ora.refine(p, oracx) * 3 + p) >> 2;
        p.clamp(1, 4095) as u32
    }

    /// Weight-set selector for each layer-1 mixer. Each is a different view of
    /// "what situation am I in", so the mixers specialise differently and layer 2
    /// learns which view to trust here.
    #[inline]
    fn mixer_selectors(&self, last: u32) -> [usize; NMIX] {
        // match state: which models are locked on, and how confidently
        let m0 = &self.matches[0];
        let m1 = &self.matches[1];
        let lenb = lenbucket(m0.len) as usize;
        let mstate = (m0.on as usize) | ((m1.on as usize) << 1) | ((self.rec_on as usize) << 2) | (lenb << 3);
        [
            self.c0 as usize,
            last as usize,
            (mstate << 3) | self.bitpos as usize,
            (self.ctxh[STR0] ^ self.c0.wrapping_mul(0x9e37_79b1)) as usize,
        ]
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        // SSE stages learn from their own cell; each mixer from its own output
        self.apm_c0.update(bit, APM_RATE);
        self.apm_o1.update(bit, APM_RATE);
        self.apm_str.update(bit, APM_RATE);
        self.apm_ora.update(bit, APM_RATE);
        for tm in &mut self.trust_match {
            tm.update(bit);
        }
        self.trust_rec.update(bit);
        self.trust_num.update(bit);
        self.mix2.update(&self.m1out, self.mix2out, bit);
        for j in 0..NMIX {
            let out = self.m1out[j];
            self.mix1[j].update(&self.st, out, bit);
        }
        // context table updates
        for m in 0..NTAB {
            let cell = unsafe { self.t.get_unchecked_mut(self.idx[m]) };
            *cell = ctr_update(*cell, bit, CLIMIT);
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
        // --- match models ---
        self.buf.push(byte);
        for m in &mut self.matches {
            m.update(&self.buf, byte);
        }

        self.stride.update(&self.buf);

        // --- record model: consume the byte before the parser can re-aim it ---
        self.rec_advance(byte);

        // --- word model: accumulate a token, retire it at the first separator ---
        if byte.is_ascii_alphanumeric() {
            // fold case so "The" and "the" share statistics
            self.word_hash = hstep(self.word_hash, byte | 0x20);
        } else if self.word_hash != 0 {
            self.prev_word = self.word_hash;
            self.word_hash = 0;
        }

        // --- structure + numeric model (format-aware) ---
        match self.mode {
            Mode::Json => self.update_struct_json(byte),
            Mode::Csv => self.update_struct_csv(byte),
            Mode::Sql => self.update_struct_sql(byte),
            Mode::Xml => self.update_struct_xml(byte),
            Mode::Generic => self.update_struct_generic(byte),
        }

        self.recompute_ctx();
    }

    /// Set up the numeric prediction for the value about to be read in `field`
    /// (column `col`, or usize::MAX for none). Chooses the more confident of two
    /// hypotheses: cross-row extrapolation (last+delta) or a cross-column relation
    /// (value = another column in this row + offset).
    fn set_np(&mut self, field: u32, col: usize) {
        let slot = self.num[(field as usize) & (NUMSLOTS - 1)];
        let ext_ok = slot.seen;
        let ext_pred = slot.last.wrapping_add(slot.delta);
        let ext_hits = if ext_ok { slot.hits } else { 0 };
        // cross-column candidate
        let mut xc_ok = false;
        let mut xc_pred = 0i64;
        let mut xc_hits = 0u32;
        if col < MAXCOL {
            let s = self.xsrc[col] as usize;
            if self.xhits[col] >= 2 && s < MAXCOL && self.row_has[s] {
                xc_pred = self.row_vals[s].wrapping_add(self.xoff[col]);
                xc_hits = self.xhits[col];
                xc_ok = true;
            }
        }
        let (pred, hits) = if xc_ok && (!ext_ok || xc_hits > ext_hits) {
            (xc_pred, xc_hits)
        } else if ext_ok {
            (ext_pred, ext_hits)
        } else {
            self.np_active = false;
            return;
        };
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
        self.num_hits = hits;
    }

    // -----------------------------------------------------------------------
    // Record-history model.
    //
    // The match model finds repeats by byte context: it needs six matching bytes
    // before it can say anything, so it is mute exactly where a new field value
    // starts. But in record-structured data the strongest predictor of a value is
    // the previous record's value *in the same field* — `"city":"Springfield"`
    // follows `"city":"Springfield"`, even though the bytes just before them
    // (a different id, a different timestamp) share nothing at all.
    //
    // So this model is a match model whose candidate is selected semantically:
    // the parser says "a value for field F starts here", and the model replays
    // what field F held last time, from its very first byte.
    // -----------------------------------------------------------------------

    #[inline]
    fn rec_slot(field: u32) -> usize {
        (field as usize) & (NUMSLOTS - 1)
    }

    /// The bit the numeric extrapolation implies, if it is still on track.
    #[inline]
    fn np_expected(&self) -> Option<u32> {
        if !self.np_active || self.np_ptr >= self.np_len {
            return None;
        }
        let pbn = self.np_digits[self.np_ptr];
        let placed = self.c0 - (1 << self.bitpos);
        if self.bitpos == 0 || placed == (pbn as u32 >> (8 - self.bitpos)) {
            Some(((pbn >> (7 - self.bitpos)) & 1) as u32)
        } else {
            None
        }
    }

    /// The bit the previous record's value implies, if this model is on track.
    #[inline]
    fn rec_expected(&self, c0: u32, bitpos: u32) -> Option<u32> {
        if !self.rec_on {
            return None;
        }
        let placed = c0 - (1 << bitpos);
        if bitpos == 0 || placed == (self.rec_pb as u32 >> (8 - bitpos)) {
            Some(((self.rec_pb >> (7 - bitpos)) & 1) as u32)
        } else {
            None
        }
    }

    /// Load the next byte to replay from the previous record's value.
    fn rec_arm(&mut self) {
        if self.rec_on && self.rec_ptr < self.rec_end && self.rec_ptr < self.buf.len() {
            self.rec_pb = self.buf[self.rec_ptr];
        } else {
            self.rec_on = false;
        }
    }

    /// A byte landed: keep replaying if it agreed with the previous record.
    fn rec_advance(&mut self, byte: u8) {
        if self.rec_on {
            if self.buf[self.rec_ptr] == byte {
                self.rec_matched += 1;
                self.rec_ptr += 1;
            } else {
                self.rec_on = false;
                let s = Self::rec_slot(self.rec_field);
                self.rec_hits[s] /= 2; // this field just proved less repetitive
            }
        }
        self.rec_arm();
    }

    /// Finish the value in progress and file it as this field's history. Called
    /// on its own where a value ends but no new one begins — the `,` between
    /// members of a JSON object, for instance, is followed by a *key*, not by
    /// another value, and folding that key into the value would blur the field.
    fn rec_close(&mut self) {
        if !self.rec_have_open {
            return;
        }
        let len = self.buf.len() as u32 - self.rec_open;
        if len > 0 && len <= REC_MAXLEN {
            let s = Self::rec_slot(self.rec_field);
            // reward on the share of the value that replayed, not on an exact
            // repeat: a timestamp that agrees for fifteen characters and then
            // diverges is still worth listening to for those fifteen
            if self.rec_matched * 2 >= len {
                self.rec_hits[s] = self.rec_hits[s].saturating_add(1);
            }
            self.rec_start[s] = self.rec_open;
            self.rec_len[s] = len;
        }
        self.rec_have_open = false;
        self.rec_on = false;
    }

    /// The parser reached a boundary: close the value in progress and open the
    /// one for `field`, pointing the model at what `field` held last record.
    fn rec_begin(&mut self, field: u32) {
        self.rec_close();
        let pos = self.buf.len() as u32;
        self.rec_field = field;
        self.rec_open = pos;
        self.rec_have_open = true;
        self.rec_matched = 0;
        let s = Self::rec_slot(field);
        if self.rec_len[s] > 0 {
            self.rec_ptr = self.rec_start[s] as usize;
            self.rec_end = self.rec_ptr + self.rec_len[s] as usize;
            self.rec_on = true;
        } else {
            self.rec_on = false;
        }
        self.rec_arm();
    }

    fn reset_row(&mut self) {
        self.row_has = [false; MAXCOL];
    }

    fn finalize_numeric(&mut self) {
        if !self.cur_is_num || self.cur_len == 0 {
            return;
        }
        let actual = if self.cur_neg { -self.cur_num } else { self.cur_num };
        // cross-row (per-field) update
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
        // cross-column update: confirm/relearn how this column relates to earlier ones
        let col = self.cur_col;
        if col < MAXCOL {
            let s = self.xsrc[col] as usize;
            let hyp_ok = self.xhits[col] >= 1
                && s < col
                && self.row_has[s]
                && self.row_vals[s].wrapping_add(self.xoff[col]) == actual;
            if hyp_ok {
                self.xhits[col] = (self.xhits[col] + 1).min(255);
            } else {
                let mut learned = false;
                for xcol in 0..col {
                    if self.row_has[xcol] && self.row_vals[xcol] == actual {
                        self.xsrc[col] = xcol as u8; // copy: value == another column
                        self.xoff[col] = 0;
                        self.xhits[col] = 1;
                        learned = true;
                        break;
                    }
                }
                if !learned {
                    if col >= 1 && self.row_has[col - 1] {
                        self.xsrc[col] = (col - 1) as u8; // offset from previous column
                        self.xoff[col] = actual.wrapping_sub(self.row_vals[col - 1]);
                        self.xhits[col] = 1;
                    } else {
                        self.xhits[col] /= 2;
                    }
                }
            }
            self.row_vals[col] = actual;
            self.row_has[col] = true;
        }
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
                    self.cur_col = usize::MAX; // JSON: no cross-column indexing
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
                self.set_np(f, usize::MAX);
                self.rec_begin(f);
                self.vpos = 0;
            }
            b'}' | b']' => {
                self.stack.pop();
                self.rec_close();
                self.vpos = 0;
            }
            b':' => {
                if let Some(f) = self.stack.last_mut() {
                    f.expect_key = false;
                }
                self.value_pending = true;
                let f = self.field_hash();
                self.set_np(f, usize::MAX);
                self.rec_begin(f);
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
                    self.rec_close(); // a key follows, not a value
                } else {
                    self.value_pending = true;
                    let f = self.field_hash();
                    self.set_np(f, usize::MAX);
                    self.rec_begin(f);
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
                // A run of whitespace is one separator. Token position within the
                // line is the closest thing to a column index unstructured text
                // has — decisive on logs, and harmlessly ignored on prose, where
                // the model never accumulates confidence and contributes nothing.
                if c == b'\n' {
                    self.gen_tok = 0;
                    self.gen_sep = true;
                    self.rec_begin(gen_field(0));
                } else if !self.gen_sep {
                    self.gen_sep = true;
                    self.gen_tok = (self.gen_tok + 1).min(255);
                    self.rec_begin(gen_field(self.gen_tok));
                }
                self.vpos = 0;
            }
            _ => {
                self.gen_sep = false;
                self.vpos = (self.vpos + 1).min(31);
            }
        }
    }

    /// SQL-dump parser: the bulk of a dump is `INSERT INTO t VALUES (..),(..)`.
    /// Each parenthesized tuple is treated like a CSV row — column index resets at
    /// `(`, increments at top-level `,`, ends at `)` — and the numeric model is
    /// routed per (column, depth), so auto-increment ids and sequential timestamps
    /// collapse. SQL string literals use single quotes (with `\` and `''` escaping).
    fn update_struct_sql(&mut self, c: u8) {
        if self.in_str {
            if self.esc {
                self.esc = false;
            } else if c == b'\\' {
                self.esc = true;
            } else if c == b'\'' {
                self.in_str = false;
            }
            self.in_num_value = false;
            self.np_active = false;
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
                    self.cur_is_num = false;
                }
                self.finalize_numeric();
                self.in_num_value = false;
                self.np_active = false;
            }
        }

        match c {
            b'\'' => {
                self.in_str = true;
                self.esc = false;
                self.sql_value_pending = false;
                self.np_active = false;
                self.vpos = 0;
            }
            b'(' => {
                if self.sql_depth == 0 {
                    self.reset_row(); // a top-level tuple is a new row
                }
                if (self.sql_depth as usize) < self.sql_col_stack.len() {
                    self.sql_col_stack[self.sql_depth as usize] = self.sql_col;
                }
                self.sql_depth = self.sql_depth.saturating_add(1).min(32);
                self.sql_col = 0;
                self.sql_value_pending = true;
                let f = sql_field(self.sql_col, self.sql_depth);
                let col = if self.sql_depth == 1 { self.sql_col as usize } else { usize::MAX };
                self.set_np(f, col);
                self.rec_begin(f);
                self.vpos = 0;
            }
            b')' => {
                self.np_active = false;
                if self.sql_depth > 0 {
                    self.sql_depth -= 1;
                    self.sql_col = self.sql_col_stack[self.sql_depth as usize];
                }
                self.sql_value_pending = false;
                self.vpos = 0;
            }
            b',' => {
                self.np_active = false;
                self.sql_col = self.sql_col.wrapping_add(1);
                self.sql_value_pending = true;
                let f = sql_field(self.sql_col, self.sql_depth);
                let col = if self.sql_depth == 1 { self.sql_col as usize } else { usize::MAX };
                self.set_np(f, col);
                self.rec_begin(f);
                self.vpos = 0;
            }
            b' ' | b'\n' | b'\r' | b'\t' => {
                self.vpos = 0;
            }
            _ => {
                if self.sql_value_pending {
                    self.sql_value_pending = false;
                    if (c.is_ascii_digit() || c == b'-') && self.sql_depth > 0 {
                        self.in_num_value = true;
                        self.cur_is_num = true;
                        self.cur_neg = c == b'-';
                        self.cur_num = 0;
                        self.cur_len = 0;
                        self.cur_field = sql_field(self.sql_col, self.sql_depth);
                        self.cur_col = if self.sql_depth == 1 { self.sql_col as usize } else { usize::MAX };
                        if c != b'-' {
                            self.cur_num = (c - b'0') as i64;
                            self.cur_len = 1;
                        }
                        self.np_consume(c);
                    } else {
                        self.in_num_value = false;
                        self.np_active = false;
                    }
                }
                self.vpos = (self.vpos + 1).min(31);
            }
        }
    }

    /// XML/HTML parser: exposes the current element tag plus parser state
    /// (in-tag / in-attribute-value / in-text) as the semantic context, so each
    /// element's content and attributes are modeled separately. Heuristic, not a
    /// validator — comments/CDATA/PIs fall through harmlessly and deterministically.
    fn update_struct_xml(&mut self, c: u8) {
        if self.xml_in_attr {
            if c == self.xml_aq {
                self.xml_in_attr = false;
            }
            self.vpos = (self.vpos + 1).min(31);
            return;
        }
        if self.xml_in_tag {
            match c {
                b'"' | b'\'' => {
                    self.xml_in_attr = true;
                    self.xml_aq = c;
                    self.vpos = 0;
                }
                b'>' => {
                    self.xml_in_tag = false;
                    if self.xml_close {
                        self.xml_stack.pop();
                    } else if !self.xml_selfclose {
                        if self.xml_stack.len() < 64 {
                            self.xml_stack.push(self.xml_cur_hash);
                        }
                    }
                    // text content of an element is that element's "field value"
                    let tag = *self.xml_stack.last().unwrap_or(&0);
                    self.rec_begin(tag ^ 0x5bf0_3635);
                    self.vpos = 0;
                }
                b'/' => {
                    if !self.xml_name_started {
                        self.xml_close = true; // </tag>
                    } else {
                        self.xml_selfclose = true; // <tag .../>
                    }
                    self.vpos = (self.vpos + 1).min(31);
                }
                b' ' | b'\n' | b'\r' | b'\t' => {
                    self.xml_reading = false; // tag name ended; attributes follow
                    self.vpos = 0;
                }
                _ => {
                    if self.xml_reading {
                        self.xml_cur_hash = hstep(self.xml_cur_hash, c);
                        self.xml_name_started = true;
                    }
                    self.vpos = (self.vpos + 1).min(31);
                }
            }
            return;
        }
        // text content between tags
        match c {
            b'<' => {
                self.xml_in_tag = true;
                self.xml_reading = true;
                self.xml_name_started = false;
                self.xml_close = false;
                self.xml_selfclose = false;
                self.xml_cur_hash = 0x9e37_79b1;
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
                    self.reset_row(); // a new line is a new row
                } else {
                    self.csv_col += 1;
                }
                let f = csv_field(self.csv_col);
                self.set_np(f, self.csv_col as usize); // prediction for the next column's value
                self.rec_begin(f);
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
                        self.cur_col = self.csv_col as usize;
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
// E8E9: x86 call/jump offset transform.
//
// `CALL` and `JMP` on x86 encode their target as an offset *relative to the
// instruction*. So the same function, called from a hundred sites, produces a
// hundred different byte sequences — invisible to every model in the portfolio.
// Rewriting each offset to an absolute address makes those hundred calls
// identical, and the match and context models light up.
//
// Reversibility is the whole game here, and it constrains the design sharply.
// Both directions must agree, position for position, on which bytes are
// instructions. The forward pass sees original bytes ahead of it; the reverse
// pass sees transformed ones. So the test may look at *only* the opcode byte,
// which neither pass ever rewrites — the four offset bytes after a hit are
// skipped by both. A tempting extra guard on the offset's sign byte (which is
// what real BCJ filters use, backed by a lookahead state machine) makes the two
// passes disagree about where instructions start, and silently corrupts data.
// ---------------------------------------------------------------------------

fn e8e9(data: &mut [u8], forward: bool) {
    let n = data.len();
    if n < 5 {
        return;
    }
    let mut i = 0;
    while i + 5 <= n {
        // 0xE8 (CALL) only: including 0xE9 (JMP) matches more sites but measures
        // worse, because unconditional jumps are rarer than the 0xE9 bytes that
        // occur by chance in data, and each false hit scrambles four bytes.
        if data[i] == 0xe8 {
            let off = i as u32 + 5;
            let rel = u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
            let v = if forward { rel.wrapping_add(off) } else { rel.wrapping_sub(off) };
            data[i + 1..i + 5].copy_from_slice(&v.to_le_bytes());
            i += 5; // the offset bytes are never themselves scanned, either way
        } else {
            i += 1;
        }
    }
}

/// Decide whether the transform pays, by trying it. Compressing a sample both
/// ways costs a fraction of a second and removes the need to guess whether a
/// file "is" an executable — which is unanswerable for an archive that merely
/// *contains* one.
fn e8e9_helps(data: &[u8], mode: Mode) -> bool {
    const SAMPLE: usize = 192 * 1024;
    if data.len() < SAMPLE * 2 {
        return false;
    }
    // The trial costs two sample encodes, which is real time to spend on a file
    // that was never going to hold machine code. Prose contains essentially no
    // 0xE8 bytes at all, while anything binary — code or not — carries them at
    // roughly 1-in-256 simply by chance. One in 2000 sits far below that and far
    // below any real instruction stream, so this only rejects text.
    let e8 = data.iter().filter(|&&b| b == 0xe8).count();
    if e8.saturating_mul(2000) < data.len() {
        return false;
    }
    // sample from the middle: archives tend to open with headers and metadata
    let start = (data.len() / 2) & !3;
    let plain = &data[start..start + SAMPLE];
    let mut xformed = plain.to_vec();
    e8e9(&mut xformed, true);
    if xformed == plain {
        return false; // no call instructions at all
    }
    let mb = mem_bits_for(SAMPLE);
    encode_stream(&xformed, mode, mb).len() < encode_stream(plain, mode, mb).len()
}

// ---------------------------------------------------------------------------
// Top-level compress / decompress
// ---------------------------------------------------------------------------

// Container layout:
//   "AUGR" | version(1) | mode(1) | mem_bits(1) | flags(1) | orig_len(8, LE) | stream
// mem_bits is stored rather than derived so the decoder builds byte-identical
// tables even if the sizing heuristic is retuned in a later release.
const MAGIC: [u8; 4] = *b"AUGR";
const VERSION: u8 = 2;
const HEADER_LEN: usize = 16;
const FLAG_E8E9: u8 = 1;

fn encode_stream(data: &[u8], mode: Mode, mem_bits: usize) -> Vec<u8> {
    let mut pr = Predictor::new(mode, mem_bits);
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

fn decode_stream(stream: &[u8], mode: Mode, mem_bits: usize, orig_len: usize) -> Vec<u8> {
    let mut pr = Predictor::new(mode, mem_bits);
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
    let mem_bits = mem_bits_for(data.len());
    // only unstructured data is a plausible carrier for machine code
    let mut flags = 0u8;
    let mut owned;
    let mut body = data;
    if mode == Mode::Generic && e8e9_helps(data, mode) {
        flags |= FLAG_E8E9;
        owned = data.to_vec();
        e8e9(&mut owned, true);
        body = &owned;
    }
    let stream = encode_stream(body, mode, mem_bits);
    let mut out = Vec::with_capacity(stream.len() + HEADER_LEN);
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(mode.to_byte());
    out.push(mem_bits as u8);
    out.push(flags);
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
    // guard the allocation: a corrupt header must not ask for terabytes
    let mem_bits = container[6] as usize;
    if !(MEM_BITS_MIN..=MEM_BITS_MAX).contains(&mem_bits) {
        return Err(format!("corrupt augur header (mem_bits {mem_bits})"));
    }
    let flags = container[7];
    let orig_len = u64::from_le_bytes(container[8..16].try_into().unwrap()) as usize;
    let mut out = decode_stream(&container[HEADER_LEN..], mode, mem_bits, orig_len);
    if flags & FLAG_E8E9 != 0 {
        e8e9(&mut out, false);
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let comp = compress(data);
        let back = decompress(&comp).expect("decompress should succeed on our own output");
        assert!(back == data, "roundtrip mismatch (len {})", data.len());
    }

    // deterministic pseudo-random bytes (no rng dependency)
    fn pseudo_random(n: usize) -> Vec<u8> {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 0xff) as u8
            })
            .collect()
    }

    #[test]
    fn empty() {
        roundtrip(b"");
    }

    #[test]
    fn one_byte() {
        roundtrip(b"A");
        roundtrip(&[0u8]);
        roundtrip(&[255u8]);
    }

    #[test]
    fn all_same_byte() {
        roundtrip(&vec![0x7e; 50_000]);
    }

    #[test]
    fn incompressible_random() {
        // must still roundtrip even though it will expand
        roundtrip(&pseudo_random(50_000));
    }

    #[test]
    fn all_byte_values() {
        let v: Vec<u8> = (0..=255u16).map(|b| b as u8).cycle().take(50_000).collect();
        roundtrip(&v);
    }

    #[test]
    fn ndjson_sequential() {
        let mut s = String::new();
        for i in 0..5_000 {
            s.push_str(&format!("{{\"id\":{},\"ts\":{},\"v\":\"x\"}}\n", 1000 + i, 1_700_000_000 + i));
        }
        roundtrip(s.as_bytes());
    }

    #[test]
    fn csv_rows() {
        let mut s = String::from("a,b,c\n");
        for i in 0..5_000 {
            s.push_str(&format!("{},{},tag{}\n", i, i * 2, i % 7));
        }
        roundtrip(s.as_bytes());
    }

    #[test]
    fn sql_dump() {
        let mut s = String::from("CREATE TABLE t (id int, ts int, name varchar(64));\n");
        for batch in 0..200 {
            s.push_str("INSERT INTO `t` VALUES ");
            for i in 0..25 {
                let id = batch * 25 + i;
                s.push_str(&format!("({},{},'it''s name{}')", 1000 + id, 1_700_000_000 + id, id % 9));
                if i < 24 { s.push(','); }
            }
            s.push_str(";\n");
        }
        roundtrip(s.as_bytes());
    }

    #[test]
    fn cross_column_csv() {
        // col1 = col0 + 500 (offset), col2 = col0 (copy), col0/col3 unpredictable
        let mut s = String::from("a,b,c,d\n");
        let mut x: u64 = 12345;
        for _ in 0..5000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = (x % 1_000_000) as i64;
            s.push_str(&format!("{},{},{},{}\n", v, v + 500, v, (x >> 20) % 97));
        }
        roundtrip(s.as_bytes());
    }

    #[test]
    fn xml_doc() {
        let mut s = String::from("<?xml version=\"1.0\"?>\n<catalog>\n");
        for i in 0..3000 {
            s.push_str(&format!(
                "  <item id=\"{}\"><name>thing {}</name><price>{}</price><tag/></item>\n",
                i, i % 50, i * 3
            ));
        }
        s.push_str("</catalog>\n");
        roundtrip(s.as_bytes());
    }

    #[test]
    fn malformed_json_is_safe() {
        // the parser is a heuristic, not a validator — must never panic or desync
        roundtrip(b"{{{not valid,,,]]] \"unterminated\n\\\\\x00\x01\xff garbage");
    }

    #[test]
    fn csv_with_quotes_and_commas() {
        roundtrip(b"\"a\",\"b,c\",\"d\"\"e\"\n1,2,3\n,,\n");
    }

    #[test]
    fn negative_and_big_numbers() {
        let mut s = String::new();
        for i in 0..2_000 {
            s.push_str(&format!("{{\"x\":{},\"y\":{}}}\n", -1000 + i, 9_000_000_000_000_000_000i64 - i as i64));
        }
        roundtrip(s.as_bytes());
    }

    #[test]
    fn e8e9_is_exactly_reversible() {
        // The pass is only safe if both directions identify the same instruction
        // positions. Adversarial input: 0xE8 bytes at every alignment, including
        // offsets whose transformed bytes are themselves 0xE8.
        let mut cases: Vec<Vec<u8>> = vec![
            b"\xe8\x00\x00\x00\x00".to_vec(),
            b"\xe8\xff\xff\xff\xff\xe8\xe8\xe8\xe8\xe8".to_vec(),
            vec![0xe8; 1000],
            (0..=255u8).cycle().take(5000).collect(),
        ];
        cases.push(pseudo_random(20_000));
        for c in cases {
            let mut x = c.clone();
            e8e9(&mut x, true);
            e8e9(&mut x, false);
            assert!(x == c, "e8e9 roundtrip mismatch (len {})", c.len());
        }
    }

    #[test]
    fn executable_like_data_roundtrips() {
        // exercises the full container path with the transform enabled
        let mut data = Vec::new();
        let mut x: u64 = 99;
        for i in 0..200_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if i % 7 == 0 {
                data.push(0xe8);
                data.extend_from_slice(&(i.wrapping_mul(31)).to_le_bytes());
            } else {
                data.push((x & 0xff) as u8);
            }
        }
        roundtrip(&data);
    }

    #[test]
    fn decompress_rejects_garbage() {
        assert!(decompress(b"").is_err());
        assert!(decompress(b"not an augur file at all").is_err());
        // valid magic, unsupported version
        let mut bad = MAGIC.to_vec();
        bad.push(99);
        bad.push(1);
        bad.extend_from_slice(&[0u8; 9]);
        assert!(decompress(&bad).is_err());
        // truncated header
        assert!(decompress(&MAGIC).is_err());
        // A corrupt mem_bits must be rejected rather than used to size an
        // allocation: 0xFF would ask for 15 tables of 2^255 counters.
        for mb in [0u8, 1, 15, 23, 64, 255] {
            let mut h = MAGIC.to_vec();
            h.push(VERSION);
            h.push(0);
            h.push(mb);
            h.push(0);
            h.extend_from_slice(&(16u64).to_le_bytes());
            assert!(decompress(&h).is_err(), "mem_bits {mb} should be rejected");
        }
    }

    #[test]
    fn header_roundtrips_every_mode() {
        // each sniffed mode must survive the container and rebuild the same parser
        let samples: Vec<Vec<u8>> = vec![
            b"{\"a\":1}\n{\"a\":2}\n".to_vec(),
            b"a,b,c\n1,2,3\n4,5,6\n7,8,9\n".to_vec(),
            b"INSERT INTO t VALUES (1,'x'),(2,'y');\n".to_vec(),
            b"<r><a>1</a></r><r><a>2</a></r>\n".to_vec(),
            b"plain text with no structure at all\n".to_vec(),
        ];
        for s in samples {
            let c = compress(&s);
            assert!(c.len() >= HEADER_LEN && c[0..4] == MAGIC);
            assert!(decompress(&c).unwrap() == s);
        }
    }
}
