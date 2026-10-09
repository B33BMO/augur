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
//!   - order 0-6, 8, 12, 16, 24 context models (local statistics), each a
//!     checksummed slot holding bit histories + direct counters, all sharing
//!     one memory pool; plus a run map, indirect, text-column and byte-class
//!     "shape" contexts
//!   - WORD models: the token being typed, with the previous word(s) as
//!     bigram, trigram and skip-gram
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
//!   - Front-ends: LMS filters for 16-bit PCM audio, a blended 2-D predictor
//!     for raw 16-bit images (geometry detected, not parsed), E8E9 for x86.
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

mod ctxbank;
mod deflate;
mod front;
mod gif;
mod jpeg;
mod recomp;
mod reflate;
mod text;
mod x86;
use front::{Front, Layout, FRONT_IN};

// ---------------------------------------------------------------------------
// Binary arithmetic coder (carryless, 32-bit). p is P(bit==1) in 16-bit units:
// on highly redundant data a 12-bit probability caps every bit at a cost of at
// least log2(4096/4095), a floor that is a visible fraction of the output.
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
        let xmid = self.x1 + ((range * p as u64) >> 16) as u32;
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
        let xmid = self.x1 + ((range * p as u64) >> 16) as u32;
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
const MEM_BITS_MAX: usize = 23;
/// All context models share one pool of 2^(mem_bits + POOL_DELTA) slots, 64
/// bytes each: 512 MB at the largest size. Sharing lets memory flow to the
/// contexts that need it — order 1 has a few thousand contexts and order 24 has
/// millions, so equal per-model tables starve one and waste the other.
const POOL_DELTA: usize = 1;
/// Slots per hash bucket: a context may live in any of them.
const WAYS: usize = 4;

fn mem_bits_for(len: usize) -> usize {
    let bl = usize::BITS - len.max(1).leading_zeros(); // ceil-ish log2
    (bl as usize + 2).clamp(MEM_BITS_MIN, MEM_BITS_MAX)
}
/// Byte-context orders modelled, in addition to order 0. Skipping 5 and 7 keeps
/// the portfolio cheap: adjacent orders are highly correlated, so the marginal
/// value of order 5 next to 4 and 6 is small compared to its table cost.
const ORDERS: [usize; 10] = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24];
const NORD: usize = 1 + ORDERS.len();
const NWORD: usize = 4; // word, (previous word, word), the trigram, and a skip-gram
const NSTR: usize = 3;
const NBIN: usize = 4; // 2 record-stride + 2 sparse contexts
const NCOL: usize = 2; // text column: the byte above in the previous line
const NIND: usize = 2; // indirect: what followed this byte / byte pair last time
const NSHAPE: usize = 3; // byte-class shape of the recent past: short, long, and fine-grained
const NTXT: usize = text::NTXT; // the text model's word, gap and line contexts
const NTAB: usize = NORD + NWORD + NSTR + NBIN + NCOL + NIND + NSHAPE + NTXT;
const WORD0: usize = NORD; // first word-model slot in ctxh
const STR0: usize = NORD + NWORD; // first structure-model slot in ctxh
const BIN0: usize = STR0 + NSTR; // first binary/record-stride slot in ctxh
/// Text contexts come last, so outside text the slot loops simply stop short.
const TXT0: usize = NTAB - NTXT;
const COL0: usize = BIN0 + NBIN; // first text-column slot in ctxh
const IND0: usize = COL0 + NCOL; // first indirect slot in ctxh
const SHAPE0: usize = IND0 + NIND;
const NMATCH: usize = 2; // match models: short-context (fast reacquire) + long (locks long repeats)
const NREC: usize = 1; // record-history model
const NRUN: usize = 1; // run map: the highest order whose context keeps repeating one byte
const ORA0: usize = 3 * NTAB; // first oracle input: each context feeds a counter and a bit history
/// Byte-history inputs: per context, the byte it saw last, trusted by how
/// often in a row it has seen it.
const BH0: usize = ORA0 + NMATCH + NREC + NRUN + 1;
const FRONT0: usize = BH0 + NTAB; // first sample front-end input
const NIN: usize = FRONT0 + FRONT_IN; // contexts + match + record + run + numeric + front-end
const MINLEN: usize = 6;
const MINLEN_LONG: usize = 16;
const CLIMIT: u16 = 15; // counter saturation: caps the slowest adaptation rate
const LR: i32 = 15; // mixer learning rate (retuned for the two-layer mixer)
const LR_SAMPLES: i32 = 8; // inside audio and image regions
const ERR_LIMIT: i32 = 12; // mixer errors (of 4096) too small to train on
const APM_RATE: i32 = 8; // SSE adaptation shift
const APM_EDGE: i32 = 1; // closest an SSE cell may get to certainty, in 1/65536
const NMIX: usize = 12; // layer-1 mixers, one per selector view

/// log2(weight sets) for each layer-1 mixer: partial byte, previous byte,
/// match-state x bit position, structure context.
/// Mixer 2's selector packs a 4-bit match-length bucket, three oracle on-flags
/// and a 3-bit bit position — ten bits. Anything narrower silently masks the
/// length bucket away, which is the most informative thing there: how much to
/// trust the match model is almost entirely a question of how long the match is.
/// Mixer 4 is keyed on the effective order (how many byte orders have seen this
/// context before), plus word and structure hits and the bit position.
const MIX_CTX_BITS: [usize; NMIX] = [10, 8, 10, 8, 9, 12, 12, 10, 12, 10, 12, 14];
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

// The hit count lives in the low 4 bits of a 16-bit counter, so a limit above 15
// would let the count overflow into the probability field and corrupt it.
const _: () = assert!(CLIMIT <= 15, "CLIMIT must fit the 4-bit count field");

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
    mask: usize,
    idx: usize,
    active: bool,
    stab: &'static [i32],
}

impl TrustMap {
    fn new() -> Self {
        Self::with_size(TRUST_CTX)
    }

    fn with_size(n: usize) -> Self {
        Self { t: vec![TRUST_INIT; n], mask: n - 1, idx: 0, active: false, stab: stretch_tab() }
    }

    /// Stretched P(bit==1) given the oracle's state; remembers the cell to train.
    #[inline]
    fn predict(&mut self, cx: usize) -> i32 {
        self.idx = cx & self.mask;
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

// The two match models have different jobs, so they search differently. The
// short-context model exists to reacquire quickly after a break, and for that the
// *most recent* occurrence is usually the right one — searching deeper finds
// candidates with more matching backward context that are nonetheless worse
// predictors, which costs real ratio on logs. The long-context model exists to
// lock onto genuine long repeats, and there depth pays enormously.
const CHAIN_SHORT: usize = 4;
const CHAIN_LONG: usize = 256;
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
    max_chain: usize,
    on: bool,
    ptr: usize,
    len: u32,
    pb: u8,
}

impl MatchModel {
    fn new(minlen: usize, max_chain: usize, mem_bits: usize) -> Self {
        Self {
            head: vec![0u32; 1 << mem_bits],
            prev: vec![0u32; 1 << mem_bits],
            mask: (1 << mem_bits) - 1,
            minlen,
            max_chain,
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
                while cand != 0 && cand < n && depth < self.max_chain {
                    let back = backmatch(buf, cand, n, MAX_BACK);
                    if back > best_back {
                        best_back = back;
                        best_pos = cand;
                        if best_back as usize >= MAX_BACK {
                            // The score saturates at MAX_BACK, so no candidate
                            // further down the chain can beat this one. Without
                            // this, a deep chain limit costs its full depth on
                            // exactly the repetitive data where the good
                            // candidate is usually the first one examined.
                            break;
                        }
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

    /// Refine a 16-bit probability; returns 16-bit. The bucket lookup only
    /// needs the 12-bit stretch domain, but the output keeps full precision.
    #[inline]
    fn refine(&mut self, pr16: i32, cx: usize) -> i32 {
        let pr = (pr16 >> 4).clamp(1, 4095);
        // SAFETY: pr is clamped to 1..4095 by every caller; s lands in 1..4095 so
        // (s>>7) is 0..31 and i+1 stays inside the context's 33-bucket row.
        let s = unsafe { *self.stab.get_unchecked(pr as usize) } + 2048;
        let w = s & 127;
        let i = (s >> 7) as usize + (cx & self.mask) * 33;
        self.idx = i + (w >> 6) as usize;
        let lo = unsafe { *self.t.get_unchecked(i) } as i32;
        let hi = unsafe { *self.t.get_unchecked(i + 1) } as i32;
        (lo * (128 - w) + hi * w) >> 7
    }

    #[inline]
    fn update(&mut self, bit: u32, rate: i32) {
        // nudge toward the observed bit, but stop short of 0/65536 so a single
        // surprise can never cost an unbounded number of bits. The target
        // overshoots by just under one step and the step truncates toward zero,
        // so a cell settles exactly APM_EDGE from either end — symmetrically.
        // (An arithmetic shift here rounds negative steps away from zero, which
        // let cells reach P=0 on the 0 side; a 12-bit output clamp used to hide it.)
        let step = 1i32 << rate;
        let g = if bit == 1 { 65535 - APM_EDGE + step - 1 } else { APM_EDGE - step + 1 };
        let cell = unsafe { self.t.get_unchecked_mut(self.idx) };
        let c = *cell as i32;
        *cell = (c + (g - c) / step).clamp(0, 65535) as u16;
    }
}

// ---------------------------------------------------------------------------
// Probability floor, chosen online.
//
// The coder takes 16-bit probabilities, so a bit can cost as little as
// log2(65536/65535). That pays handsomely on highly redundant data and hurts on
// prose, where the rare surprise after an overconfident run costs more than the
// certainty saved. No calibration stage fixes this — the surprises are not
// foreseeable from any context — so instead both sides keep a decayed tally of
// what each floor *would* have cost on the bits already coded, and use
// whichever has been cheaper. It is causal, so the decoder makes the identical
// choice with no side information, and it can change its mind mid-file.
// ---------------------------------------------------------------------------

const FLOOR_LO: i32 = 1; // full 16-bit range
const FLOOR_HI: i32 = 16; // the old 12-bit limit, 1/4096
const FLOOR_DECAY: i32 = 12; // tally half-life, in bits, is about 2^FLOOR_DECAY

/// -log2(p / 65536) in 1/65536-bit units, for p in 1..65535. The fine unit
/// matters: the savings being tallied are thousandths of a bit.
///
/// Computed in exact integer arithmetic rather than with `f64::log2`: this table
/// steers a decision the decoder must reproduce bit for bit, and libm results
/// may differ in the last place between platforms.
fn cost_tab() -> &'static [i32] {
    static T: OnceLock<Vec<i32>> = OnceLock::new();
    T.get_or_init(|| (0..65536u32).map(|p| (16 << 16) - log2_fx16(p.max(1))).collect())
}

/// log2(x) in 16.16 fixed point, by repeated squaring of the normalised
/// mantissa — integer-only, so identical on every platform.
fn log2_fx16(x: u32) -> i32 {
    let ip = 31 - x.leading_zeros() as i32;
    // mantissa in [1, 2) as 2.30 fixed point
    let mut m: u64 = ((x as u64) << 30) >> ip;
    let mut frac = 0i32;
    for bitv in (0..16).rev() {
        m = (m * m) >> 30;
        if m >= 2u64 << 30 {
            m >>= 1;
            frac |= 1 << bitv;
        }
    }
    (ip << 16) | frac
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

    /// Blend the first `live` inputs of `st` under weight set `ctx`; returns
    /// the stretched prediction. Inputs past `live` are known to be zero.
    #[inline]
    fn mix(&mut self, st: &[i32; N], ctx: usize, live: usize) -> i32 {
        self.sel = (ctx & self.mask) * N;
        // SAFETY: sel + N <= (mask+1)*N = w.len(), and live <= N
        let w = unsafe { self.w.get_unchecked(self.sel..self.sel + live) };
        let mut dot: i64 = 0;
        for i in 0..live {
            dot += w[i] as i64 * st[i] as i64;
        }
        ((dot >> 16) as i32).clamp(ST_MIN, ST_MAX)
    }

    /// Gradient step on the weight set that produced `out`.
    #[inline]
    fn update(&mut self, st: &[i32; N], out: i32, bit: u32, live: usize, lr: i32) {
        let sq = unsafe { *self.sqtab.get_unchecked((out.clamp(ST_MIN, ST_MAX) + 2048) as usize) };
        let e0 = ((bit as i32) << 12) - sq;
        // no training on errors this small: across many correlated inputs the
        // tiny updates only accumulate noise in the weights (swept 8-32)
        if e0.abs() < ERR_LIMIT {
            return;
        }
        let err = e0 * lr;
        // SAFETY: sel was set by the matching mix() call; live <= N
        let w = unsafe { self.w.get_unchecked_mut(self.sel..self.sel + live) };
        for i in 0..live {
            w[i] += (st[i] * err) >> 16;
        }
    }
}

// ---------------------------------------------------------------------------
// Bit histories — indirect context modelling.
//
// A direct counter answers "what fraction of bits were 1 here", which is the
// wrong question for data whose statistics drift: a context that saw 0000 and
// then 11 is far more likely to emit a 1 next than its 1/3 average suggests.
// A bit history keeps the *shape* of recent evidence instead — a small state
// encoding (n0, n1, last bit) under a nonstationary update that discounts the
// opposing count whenever a bit arrives — and a StateMap learns, per model, what
// each shape actually predicts on this file. That learned indirection is the
// core of what makes paq/zpaq strong on prose and binaries.
//
// The state set is generated, not tabulated: a breadth-first walk from (0,0)
// under the update rule enumerates every reachable state, which comes to well
// under 256, so a state fits in a byte.
// ---------------------------------------------------------------------------

pub(crate) struct StateTab {
    next: [[u8; 2]; 256],
    n0: [u8; 256],
    n1: [u8; 256],
}

/// Highest count a state may hold on one side, given the count on the other.
/// Lopsided histories get the most resolution: "seen forty 1s, no 0s" is a
/// common and very sharp situation, and two large counts never coexist under
/// the discounting rule anyway.
fn hist_cap(other: u32) -> u32 {
    const LIM: [u32; 9] = [48, 24, 12, 8, 6, 5, 4, 4, 3];
    LIM[(other as usize).min(LIM.len() - 1)]
}

/// Nonstationary discount of the count opposing a fresh bit.
fn hist_discount(n: u32) -> u32 {
    if n <= 2 {
        n
    } else if n < 8 {
        2 + (n - 2) / 2
    } else {
        n / 3 + 1
    }
}

fn hist_next(s: (u32, u32, u32), y: u32) -> (u32, u32, u32) {
    let (mut n0, mut n1, _) = s;
    if y == 1 {
        n1 = (n1 + 1).min(hist_cap(n0));
        n0 = hist_discount(n0);
    } else {
        n0 = (n0 + 1).min(hist_cap(n1));
        n1 = hist_discount(n1);
    }
    // the last bit only carries information while both sides are small
    let last = if n0 > 0 && n1 > 0 && n0 + n1 <= 8 { y } else { 0 };
    (n0, n1, last)
}

pub(crate) fn state_tab() -> &'static StateTab {
    static T: OnceLock<StateTab> = OnceLock::new();
    T.get_or_init(|| {
        let mut ids: Vec<(u32, u32, u32)> = vec![(0, 0, 0)];
        let mut next = [[0u8; 2]; 256];
        let mut i = 0;
        while i < ids.len() {
            for y in 0..2 {
                let t = hist_next(ids[i], y);
                let j = match ids.iter().position(|&s| s == t) {
                    Some(j) => j,
                    None => {
                        ids.push(t);
                        ids.len() - 1
                    }
                };
                next[i][y as usize] = j as u8;
            }
            i += 1;
        }
        assert!(ids.len() <= 256, "bit-history state set overflowed a byte");
        let mut t = StateTab { next, n0: [0; 256], n1: [0; 256] };
        for (k, &(a, b, _)) in ids.iter().enumerate() {
            t.n0[k] = a as u8;
            t.n1[k] = b as u8;
        }
        t
    })
}

/// state -> learned P(1), one row per model. Wide slots with a long count: what a
/// given history predicts is a property of the file, so it should settle.
const SM_LIMIT: u32 = 1023;

pub(crate) fn sm_init(st: &StateTab) -> Vec<u32> {
    (0..256)
        .map(|s| {
            let (n0, n1) = (st.n0[s] as u64, st.n1[s] as u64);
            let p22 = (((2 * n1 + 1) << 22) / (2 * (n0 + n1) + 2)) as u32;
            p22.min((1 << 22) - 1) << 10
        })
        .collect()
}

#[inline]
pub(crate) fn sm_update(v: u32, bit: u32) -> u32 {
    let n = v & 1023;
    let p22 = (v >> 10) as i32;
    let rate = unsafe { *RATE_TAB.get_unchecked(n as usize) };
    let err = (((bit as i32) << 22) - p22) as i64;
    let p22 = (p22 + ((err * rate as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
    let n = if n < SM_LIMIT { n + 1 } else { n };
    (p22 << 10) | n
}

// ---------------------------------------------------------------------------
// Context slots.
//
// One slot holds everything a context knows about one *nibble*: the fifteen
// nodes of the 4-bit binary tree, each with a direct counter and a bit history,
// behind a 16-bit checksum. Two consequences:
//
//   - a collision is detected instead of silently blending two contexts'
//     statistics, and the slot with less evidence is the one evicted;
//   - a model touches one cache line per nibble rather than one per bit.
//
// A slot is exactly one 64-byte line. Lookups are two-way: a context may live
// in either of an adjacent pair, which the hardware prefetcher usually pulls in
// together.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Slot {
    chk: u16,
    st: [u8; 15],
    _pad: u8,
    ctr: [u16; 15],
    // run map (byte-start slots only): the last byte seen in this context, and
    // how many times in a row it has been that byte
    rb: u8,
    rn: u8,
    _spare: [u8; 14],
}

const SLOT_EMPTY: Slot =
    Slot { chk: 0, st: [0; 15], _pad: 0, ctr: [CTR_INIT; 15], rb: 0, rn: 0, _spare: [0; 14] };

#[inline]
fn slot_hash(ctx: u32, nib: u32, m: usize) -> u64 {
    // splitmix64 finaliser: index from the high bits, checksum from the low,
    // so the two are effectively independent
    let mut x = ((ctx as u64) << 32) ^ ((nib as u64) << 8) ^ (m as u64) ^ 0x9e37_79b9_7f4a_7c15;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

struct Predictor {
    buf: Vec<u8>,
    slots: Vec<Slot>, // one pool shared by every context model
    slot_bits: usize,
    cur: [usize; NTAB], // slot each model is reading for the current nibble
    node: usize,        // node of the nibble tree for the current bit (0..15)
    seen: u64,          // bit m set: model m's slot already existed (not freshly claimed)
    tl: usize,          // context models live: all of them in text, the rest elsewhere
    cur0: [usize; NTAB], // byte-start slot per model, where its run map lives
    chk0: [u16; NTAB],
    run_byte: u8,
    run_cx: u32, // (order rank, run length bucket), or u32::MAX when no run applies
    trust_run: TrustMap,
    bh_byte: [u8; NTAB],   // per context: the byte it last saw (at this byte's start)
    bh_run: [u8; NTAB],    // and how many times in a row; 0 = none
    bh_tab: Vec<u32>,      // (model, run, bit position, expected bit) -> P(right)
    bh_idx: [u32; NTAB],   // cell used this bit, u32::MAX if idle
    sm: Vec<u32>,       // StateMaps: NTAB rows of 256
    stt: &'static StateTab,
    // SSE stages, applied to the mixer output in sequence
    apm_c0: Apm,
    apm_o1: Apm,
    apm_str: Apm,
    apm_ora: Apm,
    floor_tally: i64, // decayed (cost with floor LO) - (cost with floor HI)
    p_raw: i32,       // the unfloored final probability, for the tally
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
    line_start: usize, // buffer offset of the current line's first byte
    img: Option<ImgGeom>, // set when coding the residuals of a raw 16-bit image
    front: Option<Box<Front>>, // numeric sample model running in lockstep (audio, images)
    pending: Vec<(usize, Layout)>, // regions still ahead, by coded start, last first
    x86: Option<Box<x86::X86Model>>, // instruction-stream model, for executables
    text: Box<text::TextModel>,      // word, gap, line and paragraph contexts
    done: Vec<Vec<i32>>,           // samples of the regions already coded, in order
    live: usize, // mixer inputs in use: the front-end's only exist when it does
    fast: bool,  // this bit is in a JPEG scan: the byte models are skipped
    lr1: i32,    // mixer learning rates, layer 1 and 2
    lr2: i32,
    ind1: Vec<u16>, // byte -> the last two bytes that followed it
    shape: u32,     // 2-bit class per recent byte: letter / digit / space / other
    fshape: u32,    // 3-bit finer class per recent byte (case, punctuation, control, high bit)
    ind2: Vec<u16>, // byte pair -> the last two bytes that followed it
    prev_line: usize,  // ... and of the line before it
    prev_word: u32,
    prev_word2: u32,
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
    st: [i32; NIN],
    mix2out: i32, // layer-2 stretched output — what the SSE chain refines
}

impl Predictor {
    fn new(mode: Mode, mem_bits: usize) -> Self {
        let mut p = Self {
            buf: Vec::new(),
            slots: vec![SLOT_EMPTY; 1 << (mem_bits + POOL_DELTA)],
            slot_bits: mem_bits + POOL_DELTA,
            cur: [0; NTAB],
            node: 0,
            seen: 0,
            tl: TXT0,
            cur0: [0; NTAB],
            chk0: [0; NTAB],
            run_byte: 0,
            run_cx: u32::MAX,
            trust_run: TrustMap::with_size(1 << 12),
            bh_byte: [0; NTAB],
            bh_run: [0; NTAB],
            bh_tab: vec![TRUST_INIT; NTAB * 16 * 8 * 2],
            bh_idx: [u32::MAX; NTAB],
            sm: (0..NTAB).flat_map(|_| sm_init(state_tab())).collect(),
            stt: state_tab(),
            apm_c0: Apm::new(8),
            apm_o1: Apm::new(16),
            apm_str: Apm::new(16),
            apm_ora: Apm::new(14),
            floor_tally: 0,
            p_raw: 32768,
            c0: 1,
            bitpos: 0,
            ctxh: [0; NTAB],
            matches: vec![
                MatchModel::new(MINLEN, CHAIN_SHORT, mem_bits),
                MatchModel::new(MINLEN_LONG, CHAIN_LONG, mem_bits),
            ],
            trust_match: (0..NMATCH).map(|_| TrustMap::new()).collect(),
            trust_rec: TrustMap::new(),
            trust_num: TrustMap::new(),
            stride: StrideDetect::new(),
            word_hash: 0,
            line_start: 0,
            img: None,
            front: None,
            pending: Vec::new(),
            x86: None,
            text: Box::new(text::TextModel::new()),
            done: Vec::new(),
            live: FRONT0,
            fast: false,
            lr1: LR,
            lr2: LR,
            ind1: vec![0; 256],
            shape: 0,
            fshape: 0,
            ind2: vec![0; 65536],
            prev_line: 0,
            prev_word: 0,
            prev_word2: 0,
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

            st: [0; NIN],
            mix2out: 0,
        };
        p.recompute_ctx();
        p.select_slots();
        p
    }

    /// Code an image residual stream: swaps the text-only contexts for
    /// neighbourhood ones. Must be called before the first bit.
    fn set_image(&mut self, img: Option<ImgGeom>) {
        if img.is_none() {
            return; // already configured; re-selecting would mark fresh slots as seen
        }
        // undo the first nibble's claims so they are made under image contexts
        for m in 0..self.tl {
            self.slots[self.cur[m]] = SLOT_EMPTY;
        }
        self.img = img;
        self.recompute_ctx();
        self.select_slots();
    }

    /// Model the stream as x86 code (set from the container's E8E9 flag).
    fn set_x86(&mut self, on: bool) {
        self.x86 = on.then(|| Box::new(x86::X86Model::new()));
    }

    /// Schedule sample regions, each (coded start, layout), in order. Must be
    /// called at a byte boundary at or before the first region's start.
    fn set_fronts(&mut self, mut regions: Vec<(usize, Layout)>) {
        regions.reverse();
        self.pending = regions;
        if self.attach_due() {
            // the next nibble's slots were claimed under the old contexts
            for m in 0..self.tl {
                self.slots[self.cur[m]] = SLOT_EMPTY;
            }
            self.recompute_ctx();
            self.select_slots();
        }
    }

    /// At a byte boundary: retire a finished region, start a due one. Returns
    /// whether a front-end was attached.
    fn attach_due(&mut self) -> bool {
        let pos = self.buf.len();
        if let Some(f) = self.front.as_deref_mut() {
            f.locate(pos, &self.buf);
            if f.finished(pos) {
                let f = self.front.take().unwrap();
                self.done.push(f.into_samples());
                self.live = FRONT0;
            }
        }
        if self.front.is_none() && self.pending.last().is_some_and(|r| r.0 == pos) {
            let (start, lay) = self.pending.pop().unwrap();
            let f = Front::new(lay, start, &self.buf);
            self.live = FRONT0 + f.live_inputs();
            self.front = Some(Box::new(f));
            return true;
        }
        false
    }

    /// Image residual contexts. A residual's size is best predicted by its
    /// neighbours' sizes — busy regions stay busy — and once the low byte is
    /// known, it pins the high byte down almost completely.
    fn image_ctx(&mut self, g: ImgGeom) {
        let n = self.buf.len();
        if n < g.parity {
            return;
        }
        let k = n - g.parity;
        let (s, phase) = (k / 2, (k & 1) as u32);
        let w = g.width;
        let x = s % w;
        let r = |i: usize| u16::from_le_bytes([self.buf[g.parity + 2 * i], self.buf[g.parity + 2 * i + 1]]) as u32;
        let q = |v: u32| 32 - v.leading_zeros();
        let rw = if x > 0 { r(s - 1) } else { 0 };
        let rww = if x > 1 { r(s - 2) } else { 0 };
        let (rn, rnw, rne, rnn) = if s >= w {
            (
                r(s - w),
                if x > 0 { r(s - w - 1) } else { 0 },
                if x + 1 < w { r(s - w + 1) } else { 0 },
                if s >= 2 * w { r(s - 2 * w) } else { 0 },
            )
        } else {
            (0, 0, 0, 0)
        };
        let lo = if phase == 1 { self.buf[n - 1] as u32 } else { 0x100 };
        let h = |tag: u32, a: u32, b: u32| {
            (tag.wrapping_mul(0x9e37_79b1) ^ a.wrapping_mul(0x85eb_ca6b) ^ b.wrapping_mul(0xc2b2_ae35))
                .wrapping_add(phase.wrapping_mul(0x27d4_eb2f))
        };
        let busy = q(rw + rn + (rnw + rne) / 2);
        let mx = q(rw.max(rn).max(rnw).max(rne));
        let c = [
            h(1, busy, lo),
            h(2, q(rw) << 5 | q(rn), lo),
            h(3, q(rne) << 10 | q(rnw) << 5 | q(rww), lo >> 2),
            h(4, (rw & 0xff) << 5 | q(rn), lo),
            h(5, ((x >> 4) as u32) << 5 | q(rn), lo >> 4),
            h(6, q(rn) << 5 | q(rnn), lo),
            h(7, mx, lo >> 4),
        ];
        let slots = [WORD0, WORD0 + 1, WORD0 + 2, WORD0 + 3, SHAPE0, SHAPE0 + 1, SHAPE0 + 2];
        for (j, &m) in slots.iter().enumerate() {
            self.ctxh[m] = c[j];
        }
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
        self.ctxh[WORD0 + 2] = self.word_hash
            ^ self.prev_word.wrapping_mul(0x2545_f491)
            ^ self.prev_word2.wrapping_mul(0x6c8e_9cf5);
        // skip-gram: the word two back, without the one between — "the ... of"
        self.ctxh[WORD0 + 3] = self.word_hash.wrapping_mul(0x9e37_79b1)
            ^ self.prev_word2.wrapping_mul(0x85eb_ca6b)
            ^ 0x5c1b_0003;
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
        // text columns: the byte directly above in the previous line, alone and
        // with the byte to our left. Indented source, aligned logs and
        // fixed-width reports repeat vertically where no byte context sees it.
        let col = n - self.line_start;
        let above = if self.prev_line + col < self.line_start {
            self.buf[self.prev_line + col] as u32
        } else {
            0x100 // previous line is shorter than this column
        };
        let colb = col.min(255) as u32;
        // shape: the class pattern of the last eight bytes, with the last byte
        // itself — "a number, then a space, then letters" is a strong frame
        self.ctxh[SHAPE0] = (self.shape & 0xffff).wrapping_mul(0x2545_f491) ^ b(1).wrapping_mul(0x9e37_79b1) ^ 0x5ea9_0001;
        self.ctxh[SHAPE0 + 1] = self.shape.wrapping_mul(0x6c8e_9cf5) ^ 0x5ea9_0002;
        self.ctxh[SHAPE0 + 2] = (self.fshape & 0xff_ffff).wrapping_mul(0x85eb_ca6b) ^ 0x5ea9_0003;
        // indirect: "after `q` the last two times came `ue`" predicts better than `q`
        let c1 = b(1);
        let c2 = c1 | b(2) << 8;
        self.ctxh[IND0] = (c1 | (self.ind1[c1 as usize] as u32) << 8).wrapping_mul(0x9e37_79b1) ^ 0x1d1d_0001;
        self.ctxh[IND0 + 1] =
            c2.wrapping_mul(0x85eb_ca6b) ^ (self.ind2[c2 as usize] as u32).wrapping_mul(0xc2b2_ae35) ^ 0x1d1d_0002;
        self.ctxh[COL0] = above.wrapping_mul(0x6c8e_9cf5) ^ colb.wrapping_mul(0x9e37_79b1) ^ 0x7a7a_0001;
        self.ctxh[COL0 + 1] = above.wrapping_mul(0x2545_f491) ^ b(1).wrapping_mul(0x85eb_ca6b) ^ 0x7a7a_0002;
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
        // field identity crossed with the two bytes just read: "inside created_at,
        // having just seen `20`" is far more specific than either half alone
        let p2 = if n >= 2 { self.buf[n - 2] as u32 } else { 0 };
        self.ctxh[STR0 + 2] =
            field.wrapping_mul(0x7feb_352d) ^ (last | (p2 << 8)).wrapping_mul(0x846c_a68b);
        self.ctxh[STR0 + 1] = field.wrapping_mul(0x9e37_79b1)
            ^ aux.wrapping_mul(0x27d4_eb2f)
            ^ last.wrapping_mul(0x1656_67b1);
        if let Some(g) = self.img {
            self.image_ctx(g);
        }
        if self.tl == NTAB {
            self.ctxh[TXT0..].copy_from_slice(&self.text.ctx);
        }
        if let Some(f) = self.front.as_deref().filter(|f| f.active()) {
            let mut c = [0u32; 8];
            if f.slot_ctx(&mut c) {
                let slots = [WORD0, WORD0 + 1, WORD0 + 2, WORD0 + 3, SHAPE0, SHAPE0 + 1, SHAPE0 + 2];
                for (j, &m) in slots.iter().enumerate() {
                    self.ctxh[m] = c[j];
                }
            }
        }
    }

    /// Analysis aid: the JPEG bit kind of the next bit, or 16 outside a scan.
    fn bit_kind(&self) -> usize {
        match self.front.as_deref() {
            Some(front::Front::Jpeg(f)) if f.data_active() => f.bit_kind(),
            _ => 16,
        }
    }

    /// Inside a JPEG scan the byte models only add noise: skip their per-bit
    /// work entirely. Both sides decide this from the same state.
    #[inline]
    fn passthrough_fast(&self) -> bool {
        self.front.as_deref().is_some_and(|f| f.mixer_sels().is_some())
    }

    #[inline]
    fn predict(&mut self) -> u32 {
        self.fast = self.passthrough_fast();
        if self.fast {
            self.st[..FRONT0].fill(0);
            for tm in &mut self.trust_match {
                tm.idle();
            }
            self.trust_rec.idle();
            self.trust_num.idle();
            self.trust_run.idle();
            return self.predict_mix();
        }
        let stab = stretch_tab();
        let node = self.node;
        for m in self.tl..NTAB {
            self.st[m] = 0;
            self.st[NTAB + m] = 0;
            self.st[2 * NTAB + m] = 0;
        }
        for m in 0..self.tl {
            // SAFETY: cur[m] < slots.len() by construction in select_slots; node < 15
            let sl = unsafe { self.slots.get_unchecked(self.cur[m]) };
            let (cv, hs) = unsafe { (*sl.ctr.get_unchecked(node), *sl.st.get_unchecked(node)) };
            let sp = unsafe { *self.sm.get_unchecked((m << 8) | hs as usize) };
            self.st[m] = unsafe { *stab.get_unchecked(ctr_p12(cv)) };
            self.st[NTAB + m] = unsafe { *stab.get_unchecked(wide_p12(sp)) };
            // deterministic histories (every bit so far agreed) get their own
            // channel, so the mixer can trust "never contradicted" separately
            let det = (self.stt.n0[hs as usize] == 0) != (self.stt.n1[hs as usize] == 0);
            self.st[2 * NTAB + m] = if det { self.st[NTAB + m] } else { 0 };
        }
        // Oracles: each names an expected bit; a TrustMap turns that into a
        // probability whose confidence was learned from how often this oracle has
        // been right in this situation, rather than assumed.
        for i in 0..NMATCH {
            self.st[ORA0 + i] = match self.matches[i].expected(self.c0, self.bitpos) {
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
        self.st[ORA0 + NMATCH] = match self.rec_expected(self.c0, self.bitpos) {
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
        self.st[ORA0 + NMATCH + NREC] = match self.np_expected() {
            Some(eb) => {
                let cx = trust_cx(lenbucket(self.num_hits), self.bitpos, eb);
                self.trust_num.predict(cx)
            }
            None => {
                self.trust_num.idle();
                0
            }
        };

        // byte histories: each context's last byte, while it still fits the
        // bits coded so far
        {
            let placed = self.c0 - (1 << self.bitpos);
            for m in self.tl..NTAB {
                self.bh_idx[m] = u32::MAX;
                self.st[BH0 + m] = 0;
            }
            for m in 0..self.tl {
                let (rb, rn) = (self.bh_byte[m] as u32, self.bh_run[m] as usize);
                if rn > 0 && (self.bitpos == 0 || placed == rb >> (8 - self.bitpos)) {
                    let eb = (rb >> (7 - self.bitpos)) & 1;
                    let ix = ((m * 16 + rn.min(15)) * 8 + self.bitpos as usize) * 2 + eb as usize;
                    self.bh_idx[m] = ix as u32;
                    self.st[BH0 + m] = stab[wide_p12(self.bh_tab[ix])];
                } else {
                    self.bh_idx[m] = u32::MAX;
                    self.st[BH0 + m] = 0;
                }
            }
        }

        // run map: same oracle shape as the match model, one byte at a time
        self.st[ORA0 + NMATCH + NREC + 1] = {
            let placed = self.c0 - (1 << self.bitpos);
            let rb = self.run_byte as u32;
            if self.run_cx != u32::MAX && (self.bitpos == 0 || placed == rb >> (8 - self.bitpos)) {
                let eb = (rb >> (7 - self.bitpos)) & 1;
                self.trust_run.predict(((self.run_cx as usize) << 4) | ((self.bitpos as usize) << 1) | eb as usize)
            } else {
                self.trust_run.idle();
                0
            }
        };

        self.predict_mix()
    }

    /// Front-end inputs, then the mixers and the SSE chain.
    #[inline]
    fn predict_mix(&mut self) -> u32 {
        // outside sample regions the front-end inputs host the x86 model
        if self.front.is_none() {
            self.live = FRONT0;
            if let Some(x) = self.x86.as_deref_mut().filter(|x| x.on) {
                x.inputs(self.c0, &mut self.st[FRONT0..]);
                self.live = FRONT0 + x86::X86_IN;
            }
        }
        let in_samples = self.front.as_deref().is_some_and(|f| f.active() && f.sample_sels(0).is_some());
        (self.lr1, self.lr2) = if in_samples {
            // ~150 inputs of a different character: a gentler rate (swept: 8
            // beat 4-24 fixed and every decay schedule tried)
            (LR_SAMPLES, LR_SAMPLES)
        } else {
            (LR, LR)
        };
        if let Some(f) = self.front.as_deref_mut() {
            if f.active() {
                f.inputs(self.c0, self.bitpos, &mut self.st[FRONT0..]);
            } else {
                self.st[FRONT0..].fill(0);
            }
        }

        // --- two-layer mixing ---
        let last = *self.buf.last().unwrap_or(&0) as u32;
        let prev2 = if self.buf.len() >= 2 { self.buf[self.buf.len() - 2] as u32 } else { 0 };
        let mut sel = self.mixer_selectors(last);
        if self.front.is_none() {
            if let Some(x) = self.x86.as_deref().filter(|x| x.on) {
                sel[4] = x.mixer_sel(self.bitpos);
            } else if self.text.on && self.mode == Mode::Generic {
                // the structure selector is constant outside structured modes
                sel[3] = self.text.mixer_sel(self.bitpos);
            }
        }
        if let Some(f) = self.front.as_deref().filter(|f| f.active()) {
            sel[2] = f.mixer_sel(self.bitpos);
            if let Some([a, b, c, d]) = f.mixer_sels().or_else(|| f.sample_sels(self.bitpos)) {
                sel[0] = a;
                sel[1] = b;
                sel[3] = c;
                sel[4] = d;
            }
        }
        for j in 0..NMIX {
            self.m1out[j] = self.mix1[j].mix(&self.st, sel[j], self.live);
        }
        self.mix2out = self.mix2.mix(&self.m1out, self.c0 as usize, NMIX);

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

        // a JPEG scan calibrates on its own state; byte contexts mean nothing there
        let (c0cx, o1cx, strcx) = match self.front.as_deref().and_then(|f| f.apm_ctxs()) {
            Some([a, b, c]) => (a, b << 8, c << 8),
            None => (self.c0 as usize, o1cx, strcx),
        };
        let p = squash(self.mix2out).clamp(1, 4095) << 4;
        let p = (self.apm_c0.refine(p, c0cx) * 3 + p) >> 2;
        let p = (self.apm_o1.refine(p, o1cx >> 8) * 3 + p) >> 2;
        let p = (self.apm_str.refine(p, strcx >> 8) * 3 + p) >> 2;
        // in a sample region the oracles are idle; calibrate on the sample state instead
        let oracx = match self.front.as_deref().filter(|f| f.active()) {
            Some(f) => f.sse_ctx(),
            None => oracx,
        };
        let p = (self.apm_ora.refine(p, oracx) * 3 + p) >> 2;
        self.p_raw = p;
        let floor = if self.floor_tally < 0 { FLOOR_LO } else { FLOOR_HI };
        p.clamp(floor, 65536 - floor) as u32
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
            {
                let eff = (self.seen >> 1 & ((1 << ORDERS.len()) - 1)).count_ones() as usize;
                let word = (self.seen >> WORD0 & 1) as usize;
                let strc = (self.seen >> (STR0 + 2) & 1) as usize;
                (eff << 5) | (word << 4) | (strc << 3) | self.bitpos as usize
            },
            // order 2: the last two bytes, hashed down
            {
                let n = self.buf.len();
                let c2 = if n >= 2 { self.buf[n - 2] as u32 } else { 0 };
                ((last | c2 << 8).wrapping_mul(0x9e37_79b1) >> 20) as usize
            },
            // the word being typed
            (self.word_hash.wrapping_mul(0x85eb_ca6b) >> 20) as usize,
            // the partial byte with the class of the last one
            (self.c0 as usize) | ((self.shape & 3) as usize) << 8,
            // the word and the one before it
            ((self.word_hash ^ self.prev_word.wrapping_mul(0x2545_f491)).wrapping_mul(0x9e37_79b1) >> 20) as usize,
            // in text: how many of the word contexts have been seen before —
            // how much the word statistics know here (paq8px's key selector)
            if self.tl == NTAB {
                let known = (self.seen >> TXT0).count_ones() as usize;
                known << 3 | self.bitpos as usize
            } else {
                self.text.mixer_sel(self.bitpos)
            },
            // how long the match is, with the partial byte
            (lenbucket(m0.len) as usize) << 8 | self.c0 as usize,
            // order 1 with the partial byte: the classic paq selector
            (last as usize) << 6 ^ (self.c0 as usize),
        ]
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        // SSE stages learn from their own cell; each mixer from its own output
        self.apm_c0.update(bit, APM_RATE);
        self.apm_o1.update(bit, APM_RATE);
        self.apm_str.update(bit, APM_RATE);
        self.apm_ora.update(bit, APM_RATE);
        {
            let ct = cost_tab();
            let cost = |f: i32| {
                let p = self.p_raw.clamp(f, 65536 - f);
                ct[(if bit == 1 { p } else { 65536 - p }) as usize]
            };
            let d = (cost(FLOOR_LO) - cost(FLOOR_HI)) as i64;
            self.floor_tally += d - (self.floor_tally >> FLOOR_DECAY);
        }
        for tm in &mut self.trust_match {
            tm.update(bit);
        }
        self.trust_rec.update(bit);
        self.trust_num.update(bit);
        self.trust_run.update(bit);
        if !self.fast {
            for m in 0..self.tl {
                let ix = self.bh_idx[m];
                if ix != u32::MAX {
                    let c = &mut self.bh_tab[ix as usize];
                    *c = wide_update(*c, bit);
                }
            }
        }
        if let Some(f) = self.front.as_deref_mut().filter(|f| f.active()) {
            f.update(bit);
        }
        if let Some(x) = self.x86.as_deref_mut() {
            x.update(bit);
        }

        self.mix2.update(&self.m1out, self.mix2out, bit, NMIX, self.lr2);
        for j in 0..NMIX {
            let out = self.m1out[j];
            self.mix1[j].update(&self.st, out, bit, self.live, self.lr1);
        }
        // context updates: the node's counter, its bit history, and the StateMap
        // cell that history was read through
        let node = self.node;
        for m in 0..if self.fast { 0 } else { self.tl } {
            let sl = unsafe { self.slots.get_unchecked_mut(self.cur[m]) };
            let c = unsafe { sl.ctr.get_unchecked_mut(node) };
            *c = ctr_update(*c, bit, CLIMIT);
            let h = unsafe { sl.st.get_unchecked_mut(node) };
            let smc = unsafe { self.sm.get_unchecked_mut((m << 8) | *h as usize) };
            *smc = sm_update(*smc, bit);
            *h = self.stt.next[*h as usize][bit as usize];
        }
        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        if self.c0 >= 256 {
            let byte = (self.c0 - 256) as u8;
            self.byte_boundary(byte);
            self.c0 = 1;
            self.bitpos = 0;
        }
        if self.bitpos == 0 || self.bitpos == 4 {
            if !self.passthrough_fast() {
                self.select_slots();
            }
        } else {
            // node index within the nibble tree: 1 + 2 + 4 + 8 nodes, heap-ordered
            let k = self.bitpos & 3;
            self.node = ((1usize << k) - 1) + (self.c0 as usize & ((1 << k) - 1));
            if k == 3 && self.bitpos == 3 {
                // the next nibble's slot is one of two; start both loads now
                self.prefetch_slots();
            }
        }
    }

    /// Nibble context for the slot lookup: none at a byte start, the high
    /// nibble (with its leading 1) halfway through.
    #[inline]
    fn slot_index(&self, m: usize, nib: u32) -> (usize, u16) {
        let h = slot_hash(self.ctxh[m], nib, m);
        // the model index is part of the hash, so models share the pool
        // without sharing contexts
        let i = ((h >> (64 - self.slot_bits)) as usize) & !(WAYS - 1);
        (i, h as u16)
    }

    fn prefetch_slots(&self) {
        let base = self.slots.as_ptr();
        for m in 0..self.tl {
            for y in 0..2 {
                let (i, _) = self.slot_index(m, (self.c0 << 1) | y);
                // SAFETY: i < slots.len(); a prefetch has no architectural effect
                unsafe { prefetch(base.add(i) as *const u16) };
            }
        }
    }

    /// Find (or claim) each model's slot for the nibble about to be coded.
    fn select_slots(&mut self) {
        let nib = if self.bitpos == 0 { 0 } else { self.c0 };
        self.node = 0;
        self.seen = 0;
        // compute every address and start every load before touching any of
        // them, so the misses overlap instead of queueing behind each compare
        let mut at = [(0usize, 0u16); NTAB];
        let base = self.slots.as_ptr();
        for m in 0..self.tl {
            at[m] = self.slot_index(m, nib);
            // SAFETY: index < slots.len(); a prefetch has no architectural effect
            unsafe { prefetch(base.add(at[m].0) as *const u16) };
        }
        for m in 0..self.tl {
            let (i, chk) = at[m];
            let pick = match (0..WAYS).find(|&k| self.slots[i + k].chk == chk) {
                Some(k) => {
                    self.seen |= 1u64 << m;
                    i + k
                }
                None => {
                    // evict the bucket member with the thinnest root history
                    let w = |sl: &Slot| self.stt.n0[sl.st[0] as usize] as u32 + self.stt.n1[sl.st[0] as usize] as u32;
                    let v = (0..WAYS).map(|k| i + k).min_by_key(|&v| w(&self.slots[v])).unwrap();
                    self.slots[v] = Slot { chk, ..SLOT_EMPTY };
                    v
                }
            };
            self.cur[m] = pick;
        }
        if self.bitpos == 0 {
            self.run_cx = u32::MAX;
            for m in 0..self.tl {
                self.cur0[m] = self.cur[m];
                self.chk0[m] = self.slots[self.cur[m]].chk;
                let sl = &self.slots[self.cur[m]];
                let known = self.seen >> m & 1 == 1;
                self.bh_byte[m] = sl.rb;
                self.bh_run[m] = if known { sl.rn } else { 0 };
            }
            // the longest byte order still repeating itself speaks for the run map
            for r in (0..ORDERS.len()).rev() {
                let m = 1 + r;
                let sl = &self.slots[self.cur[m]];
                if self.seen >> m & 1 == 1 && sl.rn > 0 {
                    self.run_byte = sl.rb;
                    self.run_cx = ((r as u32) << 4) | lenbucket(sl.rn as u32).min(15);
                    break;
                }
            }
        }
    }

    /// Fold the byte just coded into every model's run map.
    fn update_runs(&mut self, byte: u8) {
        for m in 0..self.tl {
            let sl = &mut self.slots[self.cur0[m]];
            if sl.chk != self.chk0[m] {
                continue; // evicted by this byte's second-nibble lookup
            }
            if sl.rn > 0 && sl.rb == byte {
                sl.rn = sl.rn.saturating_add(1);
            } else {
                sl.rb = byte;
                sl.rn = 1;
            }
        }
    }

    fn byte_boundary(&mut self, byte: u8) {
        self.update_runs(byte);
        let class = if byte.is_ascii_alphabetic() {
            0
        } else if byte.is_ascii_digit() {
            1
        } else if byte == b' ' || byte == b'\n' || byte == b'\t' {
            2
        } else {
            3
        };
        self.shape = (self.shape << 2) | class;
        let fine = match byte {
            b'a'..=b'z' => 0,
            b'A'..=b'Z' => 1,
            b'0'..=b'9' => 2,
            b' ' => 3,
            b'\n' | b'\r' | b'\t' => 4,
            0x21..=0x2f | 0x3a..=0x40 | 0x5b..=0x60 | 0x7b..=0x7e => 5,
            0x80..=0xff => 6,
            _ => 7, // other control bytes, NUL included
        };
        self.fshape = (self.fshape << 3) | fine;
        // --- indirect histories: record what followed the last one and two bytes ---
        {
            let n = self.buf.len();
            if n >= 1 {
                let c1 = self.buf[n - 1] as usize;
                self.ind1[c1] = (self.ind1[c1] << 8) | byte as u16;
                if n >= 2 {
                    let c2 = c1 | (self.buf[n - 2] as usize) << 8;
                    self.ind2[c2] = (self.ind2[c2] << 8) | byte as u16;
                }
            }
        }
        // --- match models ---
        self.buf.push(byte);
        for m in &mut self.matches {
            m.update(&self.buf, byte);
        }

        self.stride.update(&self.buf);

        // --- record model: consume the byte before the parser can re-aim it ---
        self.rec_advance(byte);

        if byte == b'\n' {
            self.prev_line = self.line_start;
            self.line_start = self.buf.len();
        }

        // --- word model: accumulate a token, retire it at the first separator ---
        if byte.is_ascii_alphanumeric() {
            // fold case so "The" and "the" share statistics
            self.word_hash = hstep(self.word_hash, byte | 0x20);
        } else if self.word_hash != 0 {
            self.prev_word2 = self.prev_word;
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

        if let Some(x) = self.x86.as_deref_mut() {
            x.byte(byte);
        }
        self.text.byte(byte, self.buf.len() - 1);
        self.tl = if self.text.on { NTAB } else { TXT0 };
        self.attach_due();
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
// WAV detection. The audio model itself lives in front.rs; all this does is
// find the sample array and describe it.
// ---------------------------------------------------------------------------

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;

/// Locate the `fmt ` and `data` chunks of an integer-PCM WAV: 8, 16, 24 or 32
/// bits, one to eight channels, plain or WAVE_FORMAT_EXTENSIBLE. Float WAVs and
/// compressed codecs fall through and are compressed as ordinary bytes.
fn wav_parse(d: &[u8]) -> Option<Layout> {
    if d.len() < 44 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return None;
    }
    let u16_at = |p: usize| u16::from_le_bytes([d[p], d[p + 1]]);
    let mut pos = 12usize;
    let mut fmt: Option<(usize, usize, usize)> = None; // channels, bits, block align
    while pos + 8 <= d.len() {
        let id = &d[pos..pos + 4];
        let sz = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().ok()?) as usize;
        if id == b"fmt " && sz >= 16 && pos + 24 <= d.len() {
            let mut tag = u16_at(pos + 8);
            let channels = u16_at(pos + 10) as usize;
            let align = u16_at(pos + 20) as usize;
            let bits = u16_at(pos + 22) as usize;
            if tag == WAVE_FORMAT_EXTENSIBLE && sz >= 40 && pos + 34 <= d.len() {
                tag = u16_at(pos + 32); // first two bytes of the subformat GUID
            }
            fmt = (tag == WAVE_FORMAT_PCM).then_some((channels, bits, align));
        }
        if id == b"data" {
            let (channels, bits, align) = fmt?;
            let width = bits.div_ceil(8);
            if !(1..=4).contains(&width) || !(1..=8).contains(&channels) || align != width * channels {
                return None;
            }
            let off = pos + 8;
            let len = sz.min(d.len().saturating_sub(off));
            // whole frames only; a ragged tail is left as ordinary bytes
            let frames = len / align;
            if frames < 16 {
                return None;
            }
            let mut lay = Layout {
                kind: front::KIND_AUDIO,
                off,
                count: frames * channels,
                width: width as u8,
                shift: 0,
                flags: if width == 1 { 0 } else { front::LAY_SIGNED },
                chans: channels as u8,
                row: 0,
                stride: 0,
            };
            lay.detect_shift(d);
            return Some(lay);
        }
        pos = pos.checked_add(8)?.checked_add(sz)?.checked_add(sz & 1)?;
    }
    None
}

// ---------------------------------------------------------------------------
// Image detection: uncompressed raster formats whose headers say exactly where
// the pixels are. The image model itself lives in front.rs.
// ---------------------------------------------------------------------------

/// Build and bounds-check an image layout. Pixel samples are unsigned in all
/// these formats.
fn image_layout(d: &[u8], off: usize, w: usize, h: usize, comps: usize, width: usize, stride: usize, be: bool) -> Option<Layout> {
    let row = w.checked_mul(comps)?;
    if w == 0 || h < 2 || row > (1 << 24) || stride < row * width || stride > (1 << 28) {
        return None;
    }
    let end = off.checked_add(h.checked_mul(stride)?)?;
    if end > d.len() || h * row < 4096 {
        return None;
    }
    let mut lay = Layout {
        kind: front::KIND_IMAGE,
        off,
        count: h * row,
        width: width as u8,
        shift: 0,
        flags: if be { front::LAY_BE } else { 0 } | if comps >= 3 { front::LAY_GDIFF } else { 0 },
        chans: comps as u8,
        row,
        stride,
    };
    lay.detect_shift(d);
    Some(lay)
}

/// Binary PGM/PPM (P5/P6), 8 or 16 bits per sample.
fn pnm_parse(d: &[u8]) -> Option<Layout> {
    if d.len() < 16 || d[0] != b'P' || !(d[1] == b'5' || d[1] == b'6') {
        return None;
    }
    let comps = if d[1] == b'6' { 3 } else { 1 };
    let mut pos = 2;
    let mut vals = [0usize; 3];
    for v in vals.iter_mut() {
        // whitespace and comments between tokens
        loop {
            match d.get(pos)? {
                b'#' => {
                    while *d.get(pos)? != b'\n' {
                        pos += 1;
                    }
                }
                c if c.is_ascii_whitespace() => pos += 1,
                _ => break,
            }
        }
        let st = pos;
        while d.get(pos)?.is_ascii_digit() {
            pos += 1;
        }
        *v = std::str::from_utf8(&d[st..pos]).ok()?.parse().ok()?;
    }
    if !d.get(pos)?.is_ascii_whitespace() {
        return None;
    }
    let [w, h, maxval] = vals;
    let width = if maxval < 256 { 1 } else if maxval < 65536 { 2 } else { return None };
    image_layout(d, pos + 1, w, h, comps, width, w * comps * width, true)
}

/// Uncompressed Windows bitmap: 24/32-bit, or 8-bit with a grey palette.
fn bmp_parse(d: &[u8]) -> Option<Layout> {
    if d.len() < 54 || &d[0..2] != b"BM" {
        return None;
    }
    let u32_at = |p: usize| u32::from_le_bytes(d[p..p + 4].try_into().unwrap());
    let off = u32_at(10) as usize;
    let w = u32_at(18) as i32;
    let h = u32_at(22) as i32;
    let bpp = u16::from_le_bytes([d[28], d[29]]) as usize;
    let compression = u32_at(30);
    if w <= 0 || h == 0 || !(compression == 0 || (compression == 3 && bpp == 32)) {
        return None;
    }
    let (w, h) = (w as usize, h.unsigned_abs() as usize);
    let comps = match bpp {
        24 => 3,
        32 => 4,
        8 => {
            // palette indices are only numbers if the palette is a grey ramp
            let pal = 14 + u32_at(14) as usize;
            if pal + 1024 > off {
                return None;
            }
            let grey = (0..256).all(|i| d[pal + 4 * i..pal + 4 * i + 3].iter().all(|&c| c as usize == i));
            if !grey {
                return None;
            }
            1
        }
        _ => return None,
    };
    let stride = (w * bpp).div_ceil(32) * 4;
    image_layout(d, off, w, h, comps, 1, stride, false)
}

/// Uncompressed Targa: truecolour (type 2) or greyscale (type 3).
fn tga_parse(d: &[u8]) -> Option<Layout> {
    if d.len() < 18 || !(d[2] == 2 || d[2] == 3) || d[1] > 1 {
        return None;
    }
    let cmap_len = u16::from_le_bytes([d[5], d[6]]) as usize;
    let cmap_bits = d[7] as usize;
    let w = u16::from_le_bytes([d[12], d[13]]) as usize;
    let h = u16::from_le_bytes([d[14], d[15]]) as usize;
    let comps = match (d[2], d[16]) {
        (2, 24) => 3,
        (2, 32) => 4,
        (3, 8) => 1,
        _ => return None,
    };
    let off = 18 + d[0] as usize + if d[1] == 1 { cmap_len * cmap_bits.div_ceil(8) } else { 0 };
    image_layout(d, off, w, h, comps, 1, w * comps, false)
}

fn image_parse(d: &[u8]) -> Option<Layout> {
    pnm_parse(d).or_else(|| bmp_parse(d)).or_else(|| tga_parse(d))
}

/// Map a signed residual into u16 with small magnitudes near zero. The value is
/// reduced mod 2^16 first, which is what makes the whole pipeline exact even
/// when a prediction overshoots the 16-bit range.
#[inline]
fn zigzag16(r: i32) -> u16 {
    let v = (r as u16) as i16;
    ((v << 1) ^ (v >> 15)) as u16
}

#[inline]
fn unzigzag16(z: u16) -> i32 {
    (((z >> 1) as i16) ^ -((z & 1) as i16)) as i32
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
// ---------------------------------------------------------------------------
// Raw 16-bit images.
//
// Medical scans, astronomy frames, sensor dumps: little-endian 16-bit samples
// in raster order, often behind a header nothing here parses. Byte contexts see
// noise — the low byte of a pixel carries most of the entropy and looks random
// on its own — but the pixel is very nearly predictable from its neighbours to
// the left and above. As with audio, augur predicts each sample from ones the
// decoder will already have, and hands the CM the residual instead.
//
// The geometry is found, not parsed: the row width is the lag at which samples
// best predict each other vertically, and the byte parity is whichever
// alignment makes adjacent samples close. A trial encode has the final say.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
struct ImgGeom {
    width: usize, // samples per row
    parity: usize, // byte offset of the first sample (0 or 1)
}

const IMG_MAX_WIDTH: usize = 8192;

fn img_detect(data: &[u8]) -> Option<ImgGeom> {
    const PROBE: usize = 1 << 17; // samples examined
    if data.len() < 4 * PROBE {
        return None;
    }
    let mid = (data.len() / 2) & !1;
    let samples = |par: usize| -> Vec<i32> {
        (0..PROBE).map(|i| u16::from_le_bytes([data[mid + par + 2 * i], data[mid + par + 2 * i + 1]]) as i32).collect()
    };
    let err = |v: &[i32], w: usize| -> u64 {
        (IMG_MAX_WIDTH..v.len()).step_by(7).map(|i| (v[i] - v[i - w]).unsigned_abs() as u64).sum()
    };
    let (e0, e1) = (err(&samples(0), 1), err(&samples(1), 1));
    // a true 16-bit stream makes one alignment far smoother than the other
    let parity = if e0 * 4 < e1 {
        0
    } else if e1 * 4 < e0 {
        1
    } else {
        return None;
    };
    let v = samples(parity);
    let eh = e0.min(e1);
    let (ev, width) = (8..=IMG_MAX_WIDTH).map(|w| (err(&v, w), w)).min()?;
    // an image row predicts about as well as the left neighbour does; a 1-D
    // signal (audio, a counter) has nothing comparable at any long lag
    if ev * 4 > eh * 5 {
        return None;
    }
    Some(ImgGeom { width, parity })
}

/// LOCO-I median edge detector: picks W or N across an edge, the planar
/// estimate W + N - NW inside smooth regions.
#[inline]
fn med_predict(w: i32, n: i32, nw: i32) -> i32 {
    let (lo, hi) = if w < n { (w, n) } else { (n, w) };
    if nw >= hi {
        lo
    } else if nw <= lo {
        hi
    } else {
        w + n - nw
    }
}

const NPRED: usize = 10;

/// Candidate predictions for sample i, from its causal neighbourhood. Missing
/// neighbours (first row/column) fall back to whatever is available.
#[inline]
fn img_candidates(px: &[i32], i: usize, w: usize) -> [i32; NPRED] {
    let x = i % w;
    let at = |k: Option<usize>, d: i32| k.map(|k| px[k]).unwrap_or(d);
    let wv = at(i.checked_sub(1).filter(|_| x > 0), 0);
    let n = at(i.checked_sub(w), wv);
    let wv = if x > 0 { wv } else { n };
    let nw = at(i.checked_sub(w + 1).filter(|_| x > 0), n);
    let ne = at(i.checked_sub(w).filter(|_| x + 1 < w).map(|k| k + 1), n);
    let ww = at(i.checked_sub(2).filter(|_| x > 1), wv);
    let nn = at(i.checked_sub(2 * w), n);
    let nne = at(i.checked_sub(2 * w).filter(|_| x + 1 < w).map(|k| k + 1), ne);
    [
        wv,
        n,
        nw,
        ne,
        med_predict(wv, n, nw),
        wv + ne - n,
        (wv + n + 1) >> 1,
        n + ne - nne,
        2 * wv - ww,
        2 * n - nn,
    ]
}

/// Replace each sample with its zigzagged prediction residual (forward), or
/// rebuild samples from residuals (inverse). Bytes before `parity` and a
/// trailing odd byte pass through untouched.
///
/// The prediction blends NPRED candidates, each weighted by the inverse square
/// of the error it made at the four causal neighbours — so near an edge the
/// predictor that respects the edge takes over, and in smooth regions the
/// planar ones do. All integer, so the decoder reproduces it exactly.
fn img_transform(data: &mut [u8], g: ImgGeom, forward: bool) {
    let n = (data.len() - g.parity) / 2;
    let w = g.width;
    let body = &mut data[g.parity..g.parity + 2 * n];
    let mut px = vec![0i32; n]; // reconstructed sample values
    // per-sample absolute error of each candidate, for the neighbours' sake
    let mut perr = vec![[0u16; NPRED]; n];
    // bias cancellation: running error sum and count per texture context
    let mut bias = vec![(0i64, 0i64); 3 * 3 * 3 * 16];
    for i in 0..n {
        let x = i % w;
        let cand = img_candidates(&px, i, w);
        // error at W, N, NW, NE (those that exist)
        let mut score = [1u64; NPRED];
        let mut nb = |k: usize, wgt: u64| {
            for p in 0..NPRED {
                score[p] += perr[k][p] as u64 * wgt;
            }
        };
        if x > 0 {
            nb(i - 1, 2);
        }
        if i >= w {
            nb(i - w, 2);
            if x > 0 {
                nb(i - w - 1, 1);
            }
            if x + 1 < w {
                nb(i - w + 1, 1);
            }
        }
        let (mut num, mut den) = (0i128, 0i128);
        for p in 0..NPRED {
            let wt = (1u128 << 60) / (score[p] as u128 * score[p] as u128);
            num += wt as i128 * cand[p] as i128;
            den += wt as i128;
        }
        let pred = ((num + den / 2) / den) as i32;
        // texture context: the sign pattern of three local gradients, and how
        // busy the neighbourhood is (the best candidate's recent error)
        let sg = |d: i32| (d.signum() + 1) as usize;
        let (wv, nv, nw, ne) = (cand[0], cand[1], cand[2], cand[3]);
        let act = (64 - score.iter().min().unwrap().leading_zeros()).min(15) as usize;
        let bcx = ((sg(nv - nw) * 3 + sg(wv - nw)) * 3 + sg(ne - nv)) * 16 + act;
        let (bs, bc) = bias[bcx];
        let pred = if bc > 0 { pred + ((bs + bs.signum() * bc / 2) / bc) as i32 } else { pred };
        let b = [body[2 * i], body[2 * i + 1]];
        let v = if forward {
            let v = u16::from_le_bytes(b) as i32;
            let z = zigzag16(v - pred);
            body[2 * i..2 * i + 2].copy_from_slice(&z.to_le_bytes());
            v
        } else {
            let v = (pred + unzigzag16(u16::from_le_bytes(b))) as u16 as i32;
            body[2 * i..2 * i + 2].copy_from_slice(&(v as u16).to_le_bytes());
            v
        };
        px[i] = v;
        {
            let e = &mut bias[bcx];
            e.0 += (v - pred) as i64;
            e.1 += 1;
            if e.1 >= 256 {
                // halve, so the correction tracks the image instead of averaging it
                e.0 /= 2;
                e.1 /= 2;
            }
        }
        for p in 0..NPRED {
            perr[i][p] = (v - cand[p]).unsigned_abs().min(65535) as u16;
        }
    }
}

/// Does the image transform pay on this file? Trial-encode a band of rows.
fn img_helps(data: &[u8], g: ImgGeom, mode: Mode) -> bool {
    const SAMPLE: usize = 384 * 1024;
    let row = 2 * g.width;
    let rows = (SAMPLE / row).max(16);
    let start = g.parity + (data.len() / 2 / row) * row;
    let end = (start + rows * row).min(data.len());
    if end - start < 8 * row {
        return false;
    }
    let plain = &data[start..end];
    let mut xf = plain.to_vec();
    img_transform(&mut xf, ImgGeom { width: g.width, parity: 0 }, true);
    let mb = mem_bits_for(plain.len());
    let g0 = Some(ImgGeom { width: g.width, parity: 0 });
    encode_stream_img(&xf, mode, mb, g0).len() < encode_stream(plain, mode, mb).len()
}

/// Code a plain buffer (trial encodes for the transform decisions).
fn encode_stream(data: &[u8], mode: Mode, mem_bits: usize) -> Vec<u8> {
    encode_stream_img(data, mode, mem_bits, None)
}

fn encode_stream_img(data: &[u8], mode: Mode, mem_bits: usize, img: Option<ImgGeom>) -> Vec<u8> {
    let mut enc = Encoder::new();
    let mut pr = Predictor::new(mode, mem_bits);
    pr.set_image(img);
    for &byte in data {
        code_byte(&mut pr, &mut enc, byte);
    }
    enc.finish()
}

/// Where coded bits go: the arithmetic coder, or (for analysis) a cost tally.
trait BitSink {
    fn bit(&mut self, bit: u32, p: u32);
    /// Called after each byte, for sinks that account per byte.
    fn byte_done(&mut self) {}
    /// Analysis sinks want to know what kind of bit is coming.
    fn wants_kind(&self) -> bool {
        false
    }
    fn kind(&mut self, _k: usize) {}
}

impl BitSink for Encoder {
    #[inline]
    fn bit(&mut self, bit: u32, p: u32) {
        self.encode(bit, p);
    }
}

/// Per-byte coding cost in bits, for the `costs` analysis command.
struct CostSink {
    cur: f64,
    out: Vec<f32>,
    by_kind: [(f64, u64); 17], // JPEG bit kinds 0..15, 16 = everything else
    kind: usize,
}

impl BitSink for CostSink {
    fn bit(&mut self, bit: u32, p: u32) {
        let p = p as f64 / 65536.0;
        let c = -(if bit == 1 { p } else { 1.0 - p }).log2();
        self.cur += c;
        self.by_kind[self.kind].0 += c;
        self.by_kind[self.kind].1 += 1;
    }
    fn wants_kind(&self) -> bool {
        true
    }
    fn kind(&mut self, k: usize) {
        self.kind = k;
    }
    fn byte_done(&mut self) {
        self.out.push(self.cur as f32);
        self.cur = 0.0;
    }
}

#[inline]
fn code_byte(pr: &mut Predictor, enc: &mut impl BitSink, byte: u8) {
    for i in (0..8).rev() {
        let bit = ((byte >> i) & 1) as u32;
        if enc.wants_kind() {
            enc.kind(pr.bit_kind());
        }
        let p = pr.predict();
        enc.bit(bit, p);
        pr.update(bit);
    }
    enc.byte_done();
}

/// Coded-stream start of each sample region: residuals can be narrower than
/// the samples (wasted bits), so later regions shift left by what earlier ones
/// saved. `base` is where the virtual stream begins in the coded stream.
fn coded_starts(lays: &[Layout], base: usize) -> Vec<(usize, Layout)> {
    let mut shrink = 0;
    lays.iter()
        .map(|l| {
            let at = base + l.off - shrink;
            shrink += l.count * (l.width as usize - l.code_bytes());
            (at, *l)
        })
        .collect()
}

fn coded_len(virt_len: usize, lays: &[Layout]) -> usize {
    virt_len - lays.iter().map(|l| l.count * (l.width as usize - l.code_bytes())).sum::<usize>()
}

/// Length of the coded prefix (recipe length + recipe) when the recipe is
/// empty, as it is for every FLAG_IMG stream.
fn empty_prefix_len() -> usize {
    4 + recomp::recipe_bytes(&[], &[]).len()
}

/// Code `prefix` (the recipe) then the virtual stream, with every sample
/// region coded as residuals through its front-end.
fn encode_into(enc: &mut impl BitSink, prefix: &[u8], virt: &[u8], mode: Mode, mem_bits: usize, img: Option<ImgGeom>, lays: &[Layout], exe: bool) {
    let mut pr = Predictor::new(mode, mem_bits);
    pr.set_image(img.map(|g| ImgGeom { parity: g.parity + prefix.len(), ..g }));
    pr.set_x86(exe || mode == Mode::Generic);
    for &byte in prefix {
        code_byte(&mut pr, enc, byte);
    }
    pr.set_fronts(coded_starts(lays, prefix.len()));
    let mut pos = 0;
    for l in lays {
        for &byte in &virt[pos..l.off] {
            code_byte(&mut pr, enc, byte);
        }
        if l.kind == front::KIND_JPEG {
            // passed through: the front-end only predicts
            for &byte in &virt[l.off..l.end()] {
                code_byte(&mut pr, enc, byte);
            }
            pos = l.end();
            continue;
        }
        let cb = l.code_bytes();
        for s in 0..l.count {
            let b = pr.front.as_ref().expect("front-end attached at its region").code(l.read(virt, s));
            for &byte in &b[..cb] {
                code_byte(&mut pr, enc, byte);
            }
        }
        for g in l.gaps() {
            for &byte in &virt[g] {
                code_byte(&mut pr, enc, byte);
            }
        }
        pos = l.end();
    }
    for &byte in &virt[pos..] {
        code_byte(&mut pr, enc, byte);
    }
}

/// Inverse of encode_into: returns (recipe, virtual stream), or an error for
/// a stream that cannot be ours.
fn decode_virt(stream: &[u8], mode: Mode, mem_bits: usize, virt_len: usize, img: Option<ImgGeom>, exe: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut pr = Predictor::new(mode, mem_bits);
    let mut dec = Decoder::new(stream);
    let mut next = |pr: &mut Predictor| -> u8 {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = pr.predict();
            let bit = dec.decode(p);
            pr.update(bit);
            byte = (byte << 1) | bit as u8;
        }
        byte
    };
    // the prefix: recipe length, then the recipe
    let mut len4 = [0u8; 4];
    // the image geometry is offset by the prefix, whose length is not known
    // yet; it only matters for FLAG_IMG streams, which carry an empty recipe
    pr.set_image(img.map(|g| ImgGeom { parity: g.parity + empty_prefix_len(), ..g }));
    pr.set_x86(exe || mode == Mode::Generic);
    for b in len4.iter_mut() {
        *b = next(&mut pr);
    }
    let rlen = u32::from_le_bytes(len4) as usize;
    if rlen > stream.len().saturating_mul(64).max(1 << 20) {
        return Err("corrupt augur stream (recipe length)".into());
    }
    let mut recipe = Vec::with_capacity(rlen);
    for _ in 0..rlen {
        recipe.push(next(&mut pr));
    }
    let (_, lays) = recomp::parse_recipe(&recipe, virt_len).ok_or("corrupt augur stream (recipe)")?;
    let base = 4 + rlen;
    pr.set_fronts(coded_starts(&lays, base));
    let total = coded_len(virt_len, &lays);
    let mut coded = Vec::with_capacity(total);
    for _ in 0..total {
        coded.push(next(&mut pr));
    }
    // reassemble: raw bytes from the coded stream, samples from the front-ends
    if let Some(f) = pr.front.take() {
        pr.done.push(f.into_samples());
    }
    let mut virt = vec![0u8; virt_len];
    let (mut vpos, mut cpos) = (0usize, 0usize);
    for (k, l) in lays.iter().enumerate() {
        let n = l.off - vpos;
        virt[vpos..l.off].copy_from_slice(&coded[cpos..cpos + n]);
        cpos += n;
        if l.kind == front::KIND_JPEG {
            virt[l.off..l.end()].copy_from_slice(&coded[cpos..cpos + l.count]);
            cpos += l.count;
            vpos = l.end();
            continue;
        }
        let samples = pr.done.get(k).ok_or("corrupt augur stream (sample region)")?;
        if samples.len() != l.count {
            return Err("corrupt augur stream (sample count)".into());
        }
        for (s, &x) in samples.iter().enumerate() {
            l.write(&mut virt, s, x as i64);
        }
        cpos += l.count * l.code_bytes();
        for g in l.gaps() {
            let n = g.len();
            virt[g].copy_from_slice(&coded[cpos..cpos + n]);
            cpos += n;
        }
        vpos = l.end();
    }
    virt[vpos..].copy_from_slice(&coded[cpos..]);
    Ok((recipe, virt))
}

// Container layout (version 5):
//   "AUGR" | version(1) | mode(1) | mem_bits(1) | flags(1) | orig_len(8) | virt_len(8)
//   [image geometry, if FLAG_IMG] | coded stream
// The coded stream begins with the recipe (its length as u32 LE, then the
// recipe) so the decoder learns the sample layouts before it reaches them.
const MAGIC: [u8; 4] = *b"AUGR";
const VERSION: u8 = 5; // 5: recompression recipes and multiple sample regions
const HEADER_LEN: usize = 24;
const FLAG_E8E9: u8 = 1;
/// Raw 16-bit image; the header is followed by width (u32 LE) and parity (u8).
const FLAG_IMG: u8 = 4;
const IMG_EXT_LEN: usize = 5;

fn compress(data: &[u8]) -> Vec<u8> {
    let mut ex = recomp::expand(data);
    // a recipe is only as good as its rebuild; never ship one that fails
    if !ex.pieces.is_empty() && recomp::rebuild(&ex.virt, &ex.pieces).as_deref() != Some(data) {
        ex = recomp::Expanded { virt: data.to_vec(), pieces: Vec::new(), layouts: Vec::new() };
    }
    let mode = sniff(&ex.virt);
    let mem_bits = mem_bits_for(ex.virt.len());
    let mut flags = 0u8;
    let mut img_geom = None;
    if ex.pieces.is_empty() && ex.layouts.is_empty() {
        if let Some(g) = img_detect(&ex.virt).filter(|&g| mode == Mode::Generic && img_helps(&ex.virt, g, mode)) {
            flags |= FLAG_IMG;
            img_transform(&mut ex.virt, g, true);
            img_geom = Some(g);
        } else if mode == Mode::Generic && e8e9_helps(&ex.virt, mode) {
            // only unstructured data is a plausible carrier for machine code
            flags |= FLAG_E8E9;
            e8e9(&mut ex.virt, true);
        }
    }
    let recipe = recomp::recipe_bytes(&ex.pieces, &ex.layouts);
    let mut prefix = (recipe.len() as u32).to_le_bytes().to_vec();
    prefix.extend_from_slice(&recipe);
    debug_assert!(img_geom.is_none() || prefix.len() == empty_prefix_len());
    let mut enc = Encoder::new();
    encode_into(&mut enc, &prefix, &ex.virt, mode, mem_bits, img_geom, &ex.layouts, flags & FLAG_E8E9 != 0);
    let stream = enc.finish();
    let mut out = Vec::with_capacity(stream.len() + HEADER_LEN);
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(mode.to_byte());
    out.push(mem_bits as u8);
    out.push(flags);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&(ex.virt.len() as u64).to_le_bytes());
    if let Some(g) = img_geom {
        out.extend_from_slice(&(g.width as u32).to_le_bytes());
        out.push(g.parity as u8);
    }
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
    let virt_len = u64::from_le_bytes(container[16..24].try_into().unwrap()) as usize;
    // a virtual stream is the original with packed streams unpacked; deflate
    // tops out near 1032:1, so anything far beyond that is corruption
    if virt_len > orig_len.saturating_mul(1100).max(1 << 20) || orig_len > virt_len.saturating_mul(1100).max(1 << 20) {
        return Err("corrupt augur header (lengths)".into());
    }
    let mut at = HEADER_LEN;
    let mut img = None;
    if flags & FLAG_IMG != 0 {
        if container.len() < HEADER_LEN + IMG_EXT_LEN {
            return Err("corrupt augur header (truncated image geometry)".into());
        }
        let width = u32::from_le_bytes(container[at..at + 4].try_into().unwrap()) as usize;
        let parity = container[at + 4] as usize;
        if width == 0 || width > IMG_MAX_WIDTH || parity > 1 {
            return Err("corrupt augur header (image geometry)".into());
        }
        img = Some(ImgGeom { width, parity });
        at += IMG_EXT_LEN;
    }
    let (recipe, mut virt) = decode_virt(&container[at..], mode, mem_bits, virt_len, img, flags & FLAG_E8E9 != 0)?;
    if let Some(g) = img {
        if virt.len() > g.parity {
            img_transform(&mut virt, g, false);
        }
    }
    if flags & FLAG_E8E9 != 0 {
        e8e9(&mut virt, false);
    }
    let (pieces, _) = recomp::parse_recipe(&recipe, virt_len).ok_or("corrupt augur stream (recipe)")?;
    let out = if pieces.is_empty() { virt } else { recomp::rebuild(&virt, &pieces).ok_or("corrupt augur stream (rebuild)")? };
    if out.len() != orig_len {
        return Err("corrupt augur stream (length mismatch)".into());
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("compress") | Some("c") => cmd_compress(&args[2..]),
        Some("decompress") | Some("d") => cmd_decompress(&args[2..]),
        Some("bench") => cmd_bench(&args[2..]),
        Some("costs") => {
            // analysis aid: per-coded-byte cost in bits (f32 LE) -> <file>.costs.
            // Costs are of the coded stream: recipe first, sample regions as residuals.
            let d = read_or_die(&args[2]);
            let ex = recomp::expand(&d);
            let recipe = recomp::recipe_bytes(&ex.pieces, &ex.layouts);
            let mut prefix = (recipe.len() as u32).to_le_bytes().to_vec();
            prefix.extend_from_slice(&recipe);
            let mut sink = CostSink { cur: 0.0, out: Vec::new(), by_kind: [(0.0, 0); 17], kind: 16 };
            let exe = e8e9_helps(&ex.virt, sniff(&ex.virt));
            let mut v = ex.virt.clone();
            if exe {
                e8e9(&mut v, true);
            }
            encode_into(&mut sink, &prefix, &v, sniff(&ex.virt), mem_bits_for(ex.virt.len()), None, &ex.layouts, exe);
            let total: f64 = sink.out.iter().map(|&c| c as f64).sum();
            eprintln!("{} coded bytes ({} prefix), {:.0} bytes of cost", sink.out.len(), prefix.len(), total / 8.0);
            let names = ["code", "sign", "magnitude", "other"];
            for (k, &(c, n)) in sink.by_kind.iter().enumerate() {
                if n > 0 && k < 16 {
                    let what = format!("{} {} {}", if k & 8 != 0 { "chroma" } else { "luma" }, if k & 4 != 0 { "DC" } else { "AC" }, names[k & 3]);
                    eprintln!("  {what:22} {n:>9} bits -> {:>9.0} bytes ({:.3} bits/bit)", c / 8.0, c / n as f64);
                }
            }
            let out: Vec<u8> = sink.out.iter().flat_map(|c| c.to_le_bytes()).collect();
            write_or_die(&format!("{}.costs", args[2]), &out);
        }
        Some("zdeflate") => {
            // test aid: augur zdeflate <level> <memlevel> <wbits> <strategy> <in> <out>
            let n = |i: usize| args[i].parse::<u8>().expect("number");
            let p = deflate::Params {
                level: n(2),
                mem_level: n(3),
                wbits: n(4),
                strategy: deflate::Strategy::from_u8(n(5)).expect("strategy"),
            };
            write_or_die(&args[7], &deflate::deflate(&read_or_die(&args[6]), p));
        }
        Some("zblocks") => {
            // analysis aid: augur zblocks <raw deflate> [level memlevel wbits strategy]
            // prints the stream's blocks, and with params, zlib's blocks for the same data
            let d = read_or_die(&args[2]);
            let r = deflate::inflate(&d, usize::MAX).expect("inflate failed");
            let show = |name: &str, s: &[u8]| {
                for (i, b) in deflate::inflate_blocks(s).unwrap().iter().enumerate().take(12) {
                    println!("{name} #{i}: type {} out {}+{} syms {} matches {}", b.btype, b.out_start, b.out_len, b.symbols, b.matches);
                }
            };
            show("orig", &d);
            if args.len() >= 7 {
                let n = |i: usize| args[i].parse::<u8>().unwrap();
                let p = deflate::Params { level: n(3), mem_level: n(4), wbits: n(5), strategy: deflate::Strategy::from_u8(n(6)).unwrap() };
                let z = deflate::deflate(&r.data, p);
                show("zlib", &z);
                let i = z.iter().zip(&d).position(|(a, b)| a != b).unwrap_or(z.len().min(d.len()));
                println!("first differing byte {i} of {} / {}", d.len(), z.len());
            }
        }
        Some("zdiff") => {
            // analysis aid: augur zdiff <raw deflate> — what a reflate diff spends
            println!("{}", reflate::stats(&read_or_die(&args[2])).unwrap_or_else(|| "not reflatable".into()));
        }
        Some("expand") => {
            // analysis aid: what recompression finds in a file
            let d = read_or_die(&args[2]);
            let t = Instant::now();
            let ex = recomp::expand(&d);
            fn walk(ps: &[recomp::Piece], depth: usize, tally: &mut [usize; 4]) {
                for p in ps {
                    match p.kind {
                        recomp::Kind::Deflate(_) => tally[0] += 1,
                        recomp::Kind::Reflate { diff_len } => {
                            tally[1] += 1;
                            tally[3] += diff_len;
                        }
                        _ => tally[2] += 1,
                    }
                    walk(&p.children, depth + 1, tally);
                }
            }
            let mut tally = [0usize; 4];
            walk(&ex.pieces, 0, &mut tally);
            let ok = recomp::rebuild(&ex.virt, &ex.pieces).as_deref() == Some(&d[..]);
            println!(
                "{}: {} -> virt {}  zlib-exact {}  reflate {} (diff {} B)  other pieces {}  sample regions {}  rebuild {}  {:.1}s",
                args[2], d.len(), ex.virt.len(), tally[0], tally[1], tally[3], tally[2], ex.layouts.len(),
                if ok { "OK" } else { "FAILED" }, t.elapsed().as_secs_f64()
            );
            if args.get(3).map(String::as_str) == Some("--layouts") {
                for l in &ex.layouts {
                    println!("layout kind {} off {} count {} width {}", l.kind, l.off, l.count, l.width);
                }
            }
        }
        Some("jcheck") => {
            // analysis aid: does the JPEG model stay in sync with every scan?
            let d = read_or_die(&args[2]);
            for l in recomp::expand(&d).layouts.iter().filter(|l| l.kind == front::KIND_JPEG) {
                println!("scan at {} ({} bytes): {}", l.off, l.count, if jpeg::follows_scan(&d, l.off, l.count) { "in sync" } else { "LOST" });
            }
        }
        Some("gifcheck") => {
            for line in gif::diagnose(&read_or_die(&args[2])) {
                println!("{line}");
            }
        }
        Some("zinflate") => {
            // test aid: augur zinflate <raw deflate in> <out>
            let r = deflate::inflate(&read_or_die(&args[2]), usize::MAX).expect("inflate failed");
            eprintln!("consumed {}", r.consumed);
            write_or_die(&args[3], &r.data);
        }
        Some("size") => {
            // tuning aid: compressed size only, no decode
            for f in &args[2..] {
                println!("{f}\t{}", compress(&read_or_die(f)).len());
            }
        }
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
    fn img_transform_is_exactly_invertible_on_adversarial_data() {
        // random samples make every prediction wrong and force wraparound
        for &(w, par) in &[(1usize, 0usize), (7, 1), (64, 0), (333, 1)] {
            let orig = pseudo_random(20_001);
            let mut t = orig.clone();
            let g = ImgGeom { width: w, parity: par };
            img_transform(&mut t, g, true);
            img_transform(&mut t, g, false);
            assert!(t == orig, "image transform not invertible (w={w}, parity={par})");
        }
    }

    #[test]
    fn synthetic_image_is_detected_and_roundtrips() {
        // 12-bit smooth image with noise behind a 101-byte header (odd parity)
        let (w, h) = (700usize, 520usize);
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut d = vec![0x5au8; 101];
        for r in 0..h {
            for c in 0..w {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = (2000.0 + 800.0 * ((r as f64) / 60.0).sin() * ((c as f64) / 45.0).cos()) as i32
                    + (x % 9) as i32
                    - 4;
                d.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        assert_eq!(img_detect(&d), Some(ImgGeom { width: w, parity: 1 }));
        let comp = compress(&d);
        assert!(comp[7] & FLAG_IMG != 0, "image transform should have been chosen");
        assert!(decompress(&comp).unwrap() == d);
    }

    #[test]
    fn integer_log2_matches_float() {
        for x in [1u32, 2, 3, 5, 100, 4095, 32768, 65535] {
            let want = (x as f64).log2() * 65536.0;
            assert!((log2_fx16(x) as f64 - want).abs() <= 2.0, "log2_fx16({x})");
        }
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

    /// A WAV holding `frames` of `chans` samples, `bits` wide, from `f(frame, chan)`.
    fn make_wav_n(chans: usize, bits: usize, frames: usize, f: impl Fn(usize, usize) -> i64) -> Vec<u8> {
        let width = bits / 8;
        let data_len = frames * chans * width;
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&(chans as u16).to_le_bytes());
        w.extend_from_slice(&44100u32.to_le_bytes());
        w.extend_from_slice(&((44100 * chans * width) as u32).to_le_bytes());
        w.extend_from_slice(&((chans * width) as u16).to_le_bytes());
        w.extend_from_slice(&(bits as u16).to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&(data_len as u32).to_le_bytes());
        for n in 0..frames {
            for c in 0..chans {
                let v = f(n, c);
                // 8-bit WAV is unsigned; wider is two's complement
                let v = if bits == 8 { v + 128 } else { v };
                w.extend_from_slice(&v.to_le_bytes()[..width]);
            }
        }
        w
    }

    fn make_wav(samples: &[(i16, i16)]) -> Vec<u8> {
        make_wav_n(2, 16, samples.len(), |n, c| if c == 0 { samples[n].0 as i64 } else { samples[n].1 as i64 })
    }

    #[test]
    fn wav_survives_adversarial_audio() {
        // Residuals are coded modulo 2^bits, so anything that makes a prediction
        // overshoot — full-scale content, sign flips, noise — must still decode.
        let mut cases: Vec<Vec<(i16, i16)>> = Vec::new();
        cases.push((0..5000).map(|i| if i % 2 == 0 { (i16::MAX, i16::MIN) } else { (i16::MIN, i16::MAX) }).collect());
        cases.push((0..5000).map(|_| (i16::MAX, i16::MIN)).collect());
        cases.push((0..5000).map(|_| (0i16, 0i16)).collect());
        let noise = pseudo_random(20_000);
        cases.push(
            noise
                .chunks(4)
                .map(|c| (i16::from_le_bytes([c[0], c[1]]), i16::from_le_bytes([c[2], c[3]])))
                .collect(),
        );
        // a loud sine, which is what real music's envelope looks like
        cases.push(
            (0..5000)
                .map(|i| {
                    let v = ((i as f64 * 0.07).sin() * 32700.0) as i16;
                    (v, -v)
                })
                .collect(),
        );
        for c in cases {
            let wav = make_wav(&c);
            assert!(wav_parse(&wav).is_some(), "test wav should be recognised");
            assert_eq!(recomp::expand(&wav).layouts.len(), 1, "audio front-end should be used");
            assert!(decompress(&compress(&wav)).unwrap() == wav, "roundtrip mismatch");
        }
    }

    #[test]
    fn every_pcm_shape_roundtrips() {
        // bit depths, channel counts, and wasted low bits (a 24-bit file holding
        // 16-bit audio) — each with full-scale swings to force wraparound
        let noise = pseudo_random(1 << 16);
        for &(chans, bits, shift) in &[(1, 8, 0), (2, 8, 0), (1, 16, 0), (2, 24, 0), (2, 24, 8), (6, 16, 0), (2, 32, 0), (3, 24, 4)] {
            let max = (1i64 << (bits - 1)) - 1;
            let wav = make_wav_n(chans, bits, 3000, |n, c| {
                let v = if n % 500 < 3 {
                    if (n + c) % 2 == 0 { max } else { -max - 1 }
                } else {
                    let s = ((n as f64 * 0.05 + c as f64).sin() * max as f64 * 0.7) as i64;
                    s + (noise[(n * chans + c) % noise.len()] as i64 - 128) * (max >> 10).max(1)
                };
                (v.clamp(-max - 1, max) >> shift) << shift
            });
            let lay = wav_parse(&wav).unwrap_or_else(|| panic!("{chans}ch {bits}-bit not recognised"));
            assert_eq!(lay.shift as usize, shift, "{chans}ch {bits}-bit: wasted bits");
            let comp = compress(&wav);
            assert!(decompress(&comp).unwrap() == wav, "{chans}ch {bits}-bit roundtrip mismatch");
        }
    }

    #[test]
    fn audio_tail_after_data_chunk_survives() {
        // metadata after the samples, and a data chunk ending mid-frame
        let mut w = make_wav_n(2, 16, 2000, |n, c| ((n * 37 + c * 11) % 2000) as i64 - 1000);
        w.extend_from_slice(b"LIST\x04\x00\x00\x00abcd");
        roundtrip(&w);
        let mut t = make_wav_n(2, 24, 2000, |n, _| (n as i64 * 977) % 70000 - 35000);
        let len = t.len();
        t.truncate(len - 4); // ragged final frame
        roundtrip(&t);
    }

    #[test]
    fn non_pcm_wav_is_left_alone() {
        let mut w = make_wav(&[(1, 2), (3, 4)]);
        w[20] = 3; // IEEE float, not PCM
        assert!(wav_parse(&w).is_none());
        roundtrip(&w);
        let mut w2 = make_wav_n(2, 16, 100, |n, _| n as i64);
        w2[32] = 3; // block align inconsistent with channels * width
        assert!(wav_parse(&w2).is_none());
        roundtrip(&w2);
    }

    /// A smooth synthetic picture with noise and a hard edge, `comps` samples per pixel.
    fn picture(w: usize, h: usize, comps: usize) -> Vec<u8> {
        let noise = pseudo_random(w * h * comps);
        let mut v = Vec::with_capacity(w * h * comps);
        for y in 0..h {
            for x in 0..w {
                for c in 0..comps {
                    let base = if x > w / 2 && y > h / 3 { 230 } else { (x + 2 * y + 40 * c) % 200 };
                    v.push((base + (noise[v.len()] as usize & 7)) as u8);
                }
            }
        }
        v
    }

    #[test]
    fn images_roundtrip_through_the_front_end() {
        let (w, h) = (101, 67); // odd width: BMP rows need padding
        // PPM and PGM
        for (magic, comps) in [("P6", 3), ("P5", 1)] {
            let mut f = format!("{magic}\n# a comment\n{w} {h}\n255\n").into_bytes();
            f.extend_from_slice(&picture(w, h, comps));
            f.extend_from_slice(b"trailing junk");
            let lay = image_parse(&f).expect("pnm should be recognised");
            assert_eq!((lay.chans as usize, lay.row), (comps, w * comps));
            assert_eq!(recomp::expand(&f).layouts.len(), 1);
            assert!(decompress(&compress(&f)).unwrap() == f, "{magic} roundtrip mismatch");
        }
        // 24-bit BMP, bottom-up, with row padding that must survive verbatim
        let stride = (w * 3).div_ceil(4) * 4;
        let mut b = vec![0u8; 54];
        b[0..2].copy_from_slice(b"BM");
        b[10..14].copy_from_slice(&54u32.to_le_bytes());
        b[14..18].copy_from_slice(&40u32.to_le_bytes());
        b[18..22].copy_from_slice(&(w as u32).to_le_bytes());
        b[22..26].copy_from_slice(&(h as u32).to_le_bytes());
        b[26] = 1;
        b[28] = 24;
        let px = picture(w, h, 3);
        for y in 0..h {
            b.extend_from_slice(&px[y * w * 3..(y + 1) * w * 3]);
            b.extend_from_slice(&[0xAB, 0xCD, 0xEF][..stride - w * 3]); // non-zero padding
        }
        let lay = bmp_parse(&b).expect("bmp should be recognised");
        assert_eq!(lay.stride, stride);
        roundtrip(&b);
        // 16-bit big-endian PGM with random samples (every prediction wrong)
        let mut f = format!("P5 {w} {h} 65535\n").into_bytes();
        f.extend_from_slice(&pseudo_random(w * h * 2));
        assert_eq!(image_parse(&f).unwrap().width, 2);
        roundtrip(&f);
    }

    #[test]
    fn truncated_images_are_not_parsed() {
        let mut f = b"P6\n100 100\n255\n".to_vec();
        f.extend_from_slice(&pseudo_random(100 * 99 * 3)); // one row short
        assert!(image_parse(&f).is_none());
        roundtrip(&f);
    }

    fn zlib_wrap(raw: &[u8], level: u8) -> Vec<u8> {
        let p = deflate::Params { level, mem_level: 8, wbits: 15, strategy: deflate::Strategy::Default };
        let mut z = vec![0x78, 0xda];
        z.extend_from_slice(&deflate::deflate(raw, p));
        z.extend_from_slice(&recomp::adler32(raw).to_be_bytes());
        z
    }

    /// A deflate encoder zlib would never produce: greedy, fixed Huffman,
    /// small blocks — forces the reflate path.
    fn odd_deflate(raw: &[u8]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        let (mut acc, mut n) = (0u64, 0u32);
        let mut send = |v: u32, len: u32, bits: &mut Vec<u8>| {
            acc |= (v as u64) << n;
            n += len;
            while n >= 8 {
                bits.push(acc as u8);
                acc >>= 8;
                n -= 8;
            }
        };
        let rev = |v: u32, len: u32| (0..len).fold(0, |r, i| r | ((v >> i) & 1) << (len - 1 - i));
        let lit = |c: usize| -> (u32, u32) {
            match c {
                0..=143 => (rev(0x30 + c as u32, 8), 8),
                144..=255 => (rev(0x190 + c as u32 - 144, 9), 9),
                256..=279 => (rev(c as u32 - 256, 7), 7),
                _ => (rev(0xc0 + c as u32 - 280, 8), 8),
            }
        };
        const DBASE: [usize; 17] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257];
        const DEXTRA: [u32; 17] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7];
        let mut i = 0;
        let starts: Vec<usize> = (0..raw.len()).step_by(700).collect();
        for (k, &start) in starts.iter().enumerate() {
            let end = (start + 700).min(raw.len());
            send((k + 1 == starts.len()) as u32, 1, &mut bits);
            send(1, 2, &mut bits);
            while i < end {
                // greedy: the nearest earlier 4-byte repeat within 300 bytes
                let d = (1..=300.min(i)).find(|&d| i + 4 <= end && raw[i - d..i - d + 4] == raw[i..i + 4]);
                if let Some(d) = d {
                    let mut l = 4;
                    while l < 10 && i + l < end && raw[i - d + l] == raw[i + l] {
                        l += 1;
                    }
                    let (c, cl) = lit(257 + l - 3);
                    send(c, cl, &mut bits); // lengths 3..10 have no extra bits
                    let dk = DBASE.iter().rposition(|&b| b <= d).unwrap();
                    send(rev(dk as u32, 5), 5, &mut bits);
                    send((d - DBASE[dk]) as u32, DEXTRA[dk], &mut bits);
                    i += l;
                } else {
                    let (c, cl) = lit(raw[i] as usize);
                    send(c, cl, &mut bits);
                    i += 1;
                }
            }
            let (c, cl) = lit(256);
            send(c, cl, &mut bits);
        }
        if n > 0 {
            bits.push(acc as u8);
        }
        bits
    }

    fn text(n: usize) -> Vec<u8> {
        let words = ["alpha ", "beta ", "gamma ", "delta\n", "epsilon ", "zeta, ", "eta "];
        let r = pseudo_random(n);
        (0..n).flat_map(|i| words[r[i] as usize % words.len()].bytes()).take(n).collect()
    }

    fn recomp_paeth(a: u8, b: u8, c: u8) -> u8 {
        let p = a as i16 + b as i16 - c as i16;
        let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
        if pa <= pb && pa <= pc {
            a
        } else if pb <= pc {
            b
        } else {
            c
        }
    }

    /// A PNG whose rows use every filter type, its IDAT split in two.
    fn png(w: usize, h: usize, color: u8, depth: u8, interlace: u8, pixels: &[u8], odd: bool) -> Vec<u8> {
        let chans = match color {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        };
        let rowbytes = (w * chans * depth as usize).div_ceil(8);
        let bpp = (chans * depth as usize).div_ceil(8);
        let mut f = Vec::new();
        for y in 0..h {
            let t = (y % 5) as u8;
            f.push(t);
            for x in 0..rowbytes {
                let a = if x >= bpp { pixels[y * rowbytes + x - bpp] } else { 0 };
                let b = if y > 0 { pixels[(y - 1) * rowbytes + x] } else { 0 };
                let c = if x >= bpp && y > 0 { pixels[(y - 1) * rowbytes + x - bpp] } else { 0 };
                let p = match t {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => ((a as u16 + b as u16) / 2) as u8,
                    _ => recomp_paeth(a, b, c),
                };
                f.push(pixels[y * rowbytes + x].wrapping_sub(p));
            }
        }
        let z = if odd {
            let mut z = vec![0x78, 0x9c];
            z.extend_from_slice(&odd_deflate(&f));
            z.extend_from_slice(&recomp::adler32(&f).to_be_bytes());
            z
        } else {
            zlib_wrap(&f, 9)
        };
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        let chunk = |ty: &[u8], data: &[u8], out: &mut Vec<u8>| {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let st = out.len();
            out.extend_from_slice(ty);
            out.extend_from_slice(data);
            let c = recomp::crc32(&out[st..]);
            out.extend_from_slice(&c.to_be_bytes());
        };
        let mut ihdr = (w as u32).to_be_bytes().to_vec();
        ihdr.extend_from_slice(&(h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[depth, color, 0, 0, interlace]);
        chunk(b"IHDR", &ihdr, &mut out);
        let cut = z.len() / 3;
        chunk(b"IDAT", &z[..cut], &mut out);
        chunk(b"IDAT", &z[cut..], &mut out);
        chunk(b"IEND", &[], &mut out);
        out
    }

    #[test]
    fn zlib_clone_matches_its_own_inflate() {
        let data = text(50_000);
        for level in 1..=9 {
            for strategy in 0..5 {
                let p = deflate::Params { level, mem_level: 8, wbits: 15, strategy: deflate::Strategy::from_u8(strategy).unwrap() };
                let z = deflate::deflate(&data, p);
                assert_eq!(deflate::inflate(&z, usize::MAX).unwrap().data, data, "L{level} S{strategy}");
            }
        }
    }

    #[test]
    fn embedded_streams_are_expanded_and_rebuilt() {
        let body = text(40_000);
        // zlib in noise, a gzip member, a zip entry, and a non-zlib stream
        let mut f = pseudo_random(3000);
        f.extend_from_slice(&zlib_wrap(&body, 6));
        f.extend_from_slice(&pseudo_random(1000));
        let mut gz = vec![0x1f, 0x8b, 8, 8, 0, 0, 0, 0, 2, 3];
        gz.extend_from_slice(b"name.txt\0");
        gz.extend_from_slice(&deflate::deflate(&body, deflate::Params { level: 9, mem_level: 8, wbits: 15, strategy: deflate::Strategy::Default }));
        gz.extend_from_slice(&recomp::crc32(&body).to_le_bytes());
        gz.extend_from_slice(&(body.len() as u32).to_le_bytes());
        f.extend_from_slice(&gz);
        let odd = odd_deflate(&body[..20_000]);
        let mut zip = b"PK\x03\x04\x14\x00\x00\x00\x08\x00".to_vec();
        zip.extend_from_slice(&[0; 4]);
        zip.extend_from_slice(&recomp::crc32(&body[..20_000]).to_le_bytes());
        zip.extend_from_slice(&(odd.len() as u32).to_le_bytes());
        zip.extend_from_slice(&20_000u32.to_le_bytes());
        zip.extend_from_slice(&5u16.to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(b"a.txt");
        zip.extend_from_slice(&odd);
        f.extend_from_slice(&zip);
        f.extend_from_slice(b"trailer");
        let ex = recomp::expand(&f);
        // the zlib and gzip streams are zlib's own; the zip entry's odd encoder
        // is reflatable, but whether unpacking it pays is the trial's call
        assert!(ex.pieces.len() >= 2, "zlib and gzip streams should be found");
        assert!(ex.pieces[..2].iter().all(|p| matches!(p.kind, recomp::Kind::Deflate(_))));
        let (raw, diff, used) = reflate::analyze(&odd, usize::MAX).expect("any valid stream is reflatable");
        assert_eq!(used, odd.len());
        assert_eq!(reflate::rebuild(&raw, &diff).unwrap(), odd);
        assert_eq!(recomp::rebuild(&ex.virt, &ex.pieces).unwrap(), f);
        let comp = compress(&f);
        // 4 KB of the input is noise; the packed text should collapse around it
        assert!(comp.len() < f.len() / 2, "unpacked text should compress well: {} of {}", comp.len(), f.len());
        assert_eq!(decompress(&comp).unwrap(), f);
    }

    #[test]
    fn pngs_of_every_kind_roundtrip() {
        let (w, h) = (61, 37);
        for &(color, depth) in &[(2u8, 8u8), (6, 8), (0, 8), (4, 8), (2, 16), (0, 16), (3, 8), (0, 1), (0, 4)] {
            let chans = match color { 0 | 3 => 1, 2 => 3, 4 => 2, _ => 4 };
            let rowbytes = (w * chans * depth as usize).div_ceil(8);
            let pic = picture(rowbytes, h, 1);
            for &(interlace, odd) in &[(0u8, false), (0, true), (1, false)] {
                let f = png(w, h, color, depth, interlace, &pic, odd);
                let ex = recomp::expand(&f);
                assert_eq!(ex.pieces.len(), 1, "color {color} depth {depth} interlace {interlace} odd {odd}: not expanded");
                let numeric = color != 3 && depth >= 8 && interlace == 0;
                assert_eq!(ex.layouts.len(), numeric as usize, "color {color} depth {depth}: image layout");
                let comp = compress(&f);
                assert_eq!(decompress(&comp).unwrap(), f, "color {color} depth {depth} interlace {interlace} odd {odd}");
            }
        }
    }

    #[test]
    fn corrupt_containers_fail_cleanly() {
        // a PNG and an audio file: every recipe kind and a sample region
        let pic = picture(64 * 3, 40, 1);
        let mut inputs: Vec<Vec<u8>> = vec![png(64, 40, 2, 8, 0, &pic, true)];
        inputs.push(make_wav_n(2, 16, 3000, |n, c| ((n * 31 + c * 7) % 4000) as i64 - 2000));
        let noise = pseudo_random(4096);
        for f in inputs {
            let comp = compress(&f);
            for k in 0..200 {
                let mut bad = comp.clone();
                let i = HEADER_LEN + (noise[k * 2] as usize * 256 + noise[k * 2 + 1] as usize) % (bad.len() - HEADER_LEN);
                bad[i] ^= 1 << (k % 8);
                // must not panic; an Ok result must at least have the right length
                if let Ok(out) = decompress(&bad) {
                    assert_eq!(out.len(), f.len());
                }
            }
            for cut in [HEADER_LEN, HEADER_LEN + 3, comp.len() / 2] {
                let _ = decompress(&comp[..cut]);
            }
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        fs::read(format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"))).expect("test fixture")
    }

    #[test]
    fn jpegs_are_modelled_and_roundtrip() {
        for name in ["baseline.jpg", "restart.jpg", "gray.jpg", "sub420.jpg"] {
            let f = fixture(name);
            let ex = recomp::expand(&f);
            assert_eq!(ex.layouts.len(), 1, "{name}: one scan region");
            let l = ex.layouts[0];
            assert_eq!(l.kind, front::KIND_JPEG);
            assert!(jpeg::follows_scan(&f, l.off, l.count), "{name}: the model lost sync with the scan");
            assert_eq!(decompress(&compress(&f)).unwrap(), f, "{name}");
        }
        // progressive JPEGs are left to the byte models, but must survive
        let p = fixture("progressive.jpg");
        assert!(recomp::expand(&p).layouts.is_empty());
        assert_eq!(decompress(&compress(&p)).unwrap(), p);
        // JPEGs inside other data, a truncated one, and one with a corrupt scan
        let mut mixed = pseudo_random(2000);
        mixed.extend_from_slice(&fixture("baseline.jpg"));
        mixed.extend_from_slice(b"between");
        mixed.extend_from_slice(&fixture("restart.jpg"));
        let cut = fixture("sub420.jpg");
        mixed.extend_from_slice(&cut[..cut.len() / 2]);
        assert_eq!(recomp::expand(&mixed).layouts.len(), 2);
        assert_eq!(decompress(&compress(&mixed)).unwrap(), mixed);
        let mut bad = fixture("baseline.jpg");
        let l = recomp::expand(&bad).layouts[0];
        for i in (l.off..l.off + l.count).step_by(7) {
            if bad[i] != 0xff && bad[i - 1] != 0xff {
                bad[i] ^= 0x5a;
            }
        }
        assert_eq!(decompress(&compress(&bad)).unwrap(), bad);
    }

    /// A GIF of `pixels` (w x h, 8-bit palette) coded with the given LZW choices.
    fn make_gif(w: usize, h: usize, pixels: &[u8], lzw: &gif::Lzw, interlaced: bool) -> Vec<u8> {
        let mut g = b"GIF89a".to_vec();
        g.extend_from_slice(&(w as u16).to_le_bytes());
        g.extend_from_slice(&(h as u16).to_le_bytes());
        g.extend_from_slice(&[0xf7, 0, 0]); // 256-colour global table
        for i in 0..256u32 {
            g.extend_from_slice(&[i as u8, (i * 7) as u8, (255 - i) as u8]);
        }
        g.extend_from_slice(&[0x21, 0xfe, 5]);
        g.extend_from_slice(b"hello");
        g.push(0);
        g.push(0x2c);
        g.extend_from_slice(&[0, 0, 0, 0]);
        g.extend_from_slice(&(w as u16).to_le_bytes());
        g.extend_from_slice(&(h as u16).to_le_bytes());
        g.push(if interlaced { 0x40 } else { 0 });
        g.push(lzw.min_size);
        let order: Vec<u8> = if interlaced {
            gif::interlace_order(h).iter().flat_map(|&r| pixels[r * w..(r + 1) * w].iter().copied()).collect()
        } else {
            pixels.to_vec()
        };
        let codes = gif::encode(&order, lzw);
        // sub-blocks as the Lzw asks; derive them from the code length
        let mut l = lzw.clone();
        if l.blocks.is_empty() {
            l.blocks = vec![255; codes.len() / 255];
            l.last_block = (codes.len() % 255) as u8;
            if l.last_block == 0 && !l.blocks.is_empty() {
                l.blocks.pop();
                l.last_block = 255;
            }
        }
        g.extend_from_slice(&gif::blocks(&codes, &l).unwrap());
        g.push(0x3b);
        g
    }

    #[test]
    fn gifs_roundtrip_through_lzw_recompression() {
        let (w, h) = (97, 61);
        // few colours in flat regions, the way GIFs are
        let pixels: Vec<u8> = picture(w, h, 1).iter().map(|&v| v / 16).collect();
        let n = (w * h) as u32;
        let base = gif::Lzw { min_size: 8, clears: vec![0], cuts: vec![], end_code: true, blocks: vec![], last_block: 0 };
        let cases = [
            base.clone(),
            gif::Lzw { clears: vec![], ..base.clone() },              // no initial clear, never clears
            gif::Lzw { clears: vec![0, n / 3, n / 2], ..base.clone() }, // early resets
            gif::Lzw { end_code: false, ..base.clone() },
            gif::Lzw { min_size: 4, ..base.clone() },
            // an encoder that cuts strings short of greedy now and then
            gif::Lzw { cuts: vec![(500, 1), (1200, 2), (3000, 1)], ..base.clone() },
        ];
        for (k, lzw) in cases.iter().enumerate() {
            for interlaced in [false, true] {
                let f = make_gif(w, h, &pixels, lzw, interlaced);
                let ex = recomp::expand(&f);
                assert_eq!(ex.pieces.len(), 1, "case {k} interlaced {interlaced}: not expanded");
                assert_eq!(ex.layouts.len(), 1);
                // the pixels are in display order whichever way the file stores them
                let l = ex.layouts[0];
                assert_eq!(&ex.virt[l.off..l.off + w * h], &pixels[..]);
                assert_eq!(decompress(&compress(&f)).unwrap(), f, "case {k} interlaced {interlaced}");
            }
        }
        // odd sub-block sizes survive
        let mut odd = base.clone();
        let codes = gif::encode(&pixels, &odd);
        odd.blocks = vec![100; codes.len() / 100];
        odd.last_block = (codes.len() % 100) as u8;
        let f = make_gif(w, h, &pixels, &odd, false);
        assert_eq!(recomp::expand(&f).pieces.len(), 1);
        assert_eq!(decompress(&compress(&f)).unwrap(), f);
        // a damaged LZW stream is left alone, but still roundtrips
        let mut bad = make_gif(w, h, &pixels, &base, false);
        let len = bad.len();
        bad[len - 40] ^= 0x55;
        assert_eq!(decompress(&compress(&bad)).unwrap(), bad);
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
        // derived from the bounds so raising the cap can't silently invalidate this
        let below = MEM_BITS_MIN as u8 - 1;
        let above = MEM_BITS_MAX as u8 + 1;
        for mb in [0u8, 1, below, above, 64, 255] {
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
