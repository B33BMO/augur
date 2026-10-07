//! Streaming sample front-end.
//!
//! Audio and raster data are numbers, not bytes: sample 44100 of a song has
//! almost nothing byte-wise in common with sample 44099 even though it is
//! nearly *numerically* equal to it. A byte-level CM sees noise.
//!
//! The front-end runs a numeric model in lockstep with the CM. Before each
//! sample it makes a main prediction; the CM codes the residual (offset binary,
//! most significant byte first) instead of the sample. The model also keeps a
//! portfolio of *other* predictors, and at every bit each of them contributes a
//! context of the form "my prediction minus what is known of this residual so
//! far" — so the CM learns, per bit, which predictor to believe in this region
//! of the signal. This is the paq8px/OptimFROG idea; the difference here is
//! that the coded symbol is already centred by the best predictor, which keeps
//! the ordinary byte models useful as well.
//!
//! Both sides reconstruct each sample from bytes already coded, so the decoder
//! rebuilds every predictor bit-identically and nothing is transmitted beyond
//! the layout. Floating point is safe for that: Rust never contracts `a*b+c`
//! into a fused multiply-add, and `+ - * / sqrt` are correctly rounded by
//! IEEE 754, so the same operation sequence gives the same bits on every
//! conforming platform. Nothing here calls a transcendental function.

use super::{sm_init, sm_update, state_tab, stretch_tab, StateTab, RATE_TAB};


// ---------------------------------------------------------------------------
// Layout: where a sample array lives inside the file.
// ---------------------------------------------------------------------------

pub const LAY_SIGNED: u8 = 1; // samples are two's complement (else offset binary, e.g. 8-bit WAV)
pub const LAY_BE: u8 = 2; // big-endian (AIFF)
/// Samples are palette indices: equal means equal, but near means nothing.
pub const LAY_PALETTE: u8 = 4;
/// Colour samples are coded as G, R-G, B-G (mod 2^bits): green carries most
/// of the luminance, and the differences are smooth chroma. Needs >= 3
/// components with green second; alpha is left alone.
pub const LAY_GDIFF: u8 = 8;

/// Model selector stored in the container, so the decoder builds the same one.
pub const KIND_AUDIO: u8 = 1;
pub const KIND_IMAGE: u8 = 2;
/// A JPEG scan's entropy-coded data, passed through and modelled bit by bit;
/// `stride` holds the distance back to the JPEG's SOI marker.
pub const KIND_JPEG: u8 = 3;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Layout {
    pub kind: u8,
    pub off: usize,   // byte offset of the first sample in the original data
    pub count: usize, // samples (all channels)
    pub width: u8,    // bytes per sample in the original, 1..=4
    pub shift: u8,    // low bits that are zero in every sample
    pub flags: u8,
    pub chans: u8,    // audio channels, or components per pixel
    pub row: usize,   // samples per raster row (images); 0 for audio
    pub stride: usize, // bytes per raster row, padding included
}

pub const LAYOUT_LEN: usize = 1 + 8 + 8 + 4 + 8;

impl Layout {
    /// Significant bits per sample, after the wasted low bits are dropped.
    pub fn bits(&self) -> u32 {
        self.width as u32 * 8 - self.shift as u32
    }
    /// Bytes per coded residual.
    pub fn code_bytes(&self) -> usize {
        (self.bits() as usize).div_ceil(8)
    }
    pub fn end(&self) -> usize {
        if self.row == 0 {
            self.off + self.count * self.width as usize
        } else {
            self.off + self.count / self.row * self.stride
        }
    }

    #[inline]
    fn pos(&self, s: usize) -> usize {
        let w = self.width as usize;
        if self.row == 0 {
            self.off + s * w
        } else {
            self.off + s / self.row * self.stride + s % self.row * w
        }
    }

    /// Byte ranges inside the sample region that hold no sample (row padding),
    /// in file order. They are coded verbatim after the residuals.
    pub fn gaps(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let rows = if self.row == 0 { 0 } else { self.count / self.row };
        let used = self.row * self.width as usize;
        (0..rows)
            .map(move |r| self.off + r * self.stride + used..self.off + (r + 1) * self.stride)
            .filter(|g| !g.is_empty())
    }

    /// Raw unsigned value of the sample stored at `pos(s)`.
    #[inline]
    fn raw(&self, d: &[u8], s: usize) -> u32 {
        let w = self.width as usize;
        let p = self.pos(s);
        let mut v: u32 = 0;
        for i in 0..w {
            let b = if self.flags & LAY_BE != 0 { d[p + i] } else { d[p + w - 1 - i] };
            v = (v << 8) | b as u32;
        }
        v
    }

    #[inline]
    fn put_raw(&self, d: &mut [u8], s: usize, v: u32) {
        let w = self.width as usize;
        let p = self.pos(s);
        for i in 0..w {
            let b = (v >> (8 * i)) as u8;
            if self.flags & LAY_BE != 0 {
                d[p + w - 1 - i] = b;
            } else {
                d[p + i] = b;
            }
        }
    }

    /// For the colour transform: (stored sample of coded sample s, and the
    /// green it is taken relative to, if any).
    #[inline]
    fn gdiff_map(&self, s: usize) -> (usize, Option<usize>) {
        let ch = self.chans as usize;
        let (px, c) = (s / ch * ch, s % ch);
        match c {
            0 => (px + 1, None),          // green first
            1 => (px, Some(px + 1)),      // red - green
            2 => (px + 2, Some(px + 1)),  // blue - green
            _ => (px + c, None),          // alpha
        }
    }

    /// Sample `s` (in coded order) as a signed value with the wasted bits removed.
    #[inline]
    pub fn read(&self, d: &[u8], s: usize) -> i64 {
        let w = self.width as usize;
        let bits = 8 * w as u32;
        let mut v = if self.flags & LAY_GDIFF != 0 {
            let (at, g) = self.gdiff_map(s);
            let v = self.raw(d, at);
            match g {
                Some(g) => v.wrapping_sub(self.raw(d, g)) & ((1u64 << bits) - 1) as u32,
                None => v,
            }
        } else {
            self.raw(d, s)
        };
        if self.flags & LAY_SIGNED == 0 {
            v ^= 1 << (bits - 1);
        }
        let x = ((v as i64) << (64 - bits)) >> (64 - bits);
        x >> self.shift
    }

    #[inline]
    /// Store coded sample `s`. With the colour transform, green must already
    /// be stored — it is, because samples are written in coded order.
    pub fn write(&self, d: &mut [u8], s: usize, x: i64) {
        let w = self.width as usize;
        let bits = 8 * w as u32;
        let mask = ((1u64 << bits) - 1) as u32;
        let mut v = ((x << self.shift) as u64 & mask as u64) as u32;
        if self.flags & LAY_SIGNED == 0 {
            v ^= 1 << (bits - 1);
        }
        if self.flags & LAY_GDIFF != 0 {
            let (at, g) = self.gdiff_map(s);
            if let Some(g) = g {
                v = v.wrapping_add(self.raw(d, g)) & mask;
            }
            self.put_raw(d, at, v);
        } else {
            self.put_raw(d, s, v);
        }
    }

    /// Wasted low bits: those zero in every sample. FLAC calls these "wasted
    /// bits"; a 24-bit file mastered from 16-bit sources has eight of them.
    pub fn detect_shift(&mut self, d: &[u8]) {
        self.shift = 0;
        let mut acc: i64 = 0;
        for s in 0..self.count {
            acc |= self.read(d, s);
            if acc & 1 != 0 {
                return;
            }
        }
        let max = self.width as u32 * 8 - 1;
        self.shift = if acc == 0 { 0 } else { (acc.trailing_zeros()).min(max) as u8 };
    }

    pub fn to_bytes(&self) -> [u8; LAYOUT_LEN] {
        let mut b = [0u8; LAYOUT_LEN];
        b[0] = self.kind;
        b[1..9].copy_from_slice(&(self.off as u64).to_le_bytes());
        b[9..17].copy_from_slice(&(self.count as u64).to_le_bytes());
        b[17] = self.width;
        b[18] = self.shift;
        b[19] = self.flags;
        b[20] = self.chans;
        b[21..25].copy_from_slice(&(self.row as u32).to_le_bytes());
        b[25..29].copy_from_slice(&(self.stride as u32).to_le_bytes());
        b
    }

    /// Parse and validate against the original length; a corrupt header must
    /// not be able to index out of bounds or allocate absurdly.
    pub fn from_bytes(b: &[u8], orig_len: usize) -> Option<Layout> {
        if b.len() < LAYOUT_LEN {
            return None;
        }
        let l = Layout {
            kind: b[0],
            off: u64::from_le_bytes(b[1..9].try_into().ok()?) as usize,
            count: u64::from_le_bytes(b[9..17].try_into().ok()?) as usize,
            width: b[17],
            shift: b[18],
            flags: b[19],
            chans: b[20],
            row: u32::from_le_bytes(b[21..25].try_into().ok()?) as usize,
            stride: u32::from_le_bytes(b[25..29].try_into().ok()?) as usize,
        };
        let w = l.width as usize;
        let shape_ok = match l.kind {
            KIND_AUDIO => l.row == 0 && l.count.checked_mul(w)?.checked_add(l.off)? <= orig_len,
            KIND_JPEG => {
                l.row == 0 && l.width == 1 && l.stride <= l.off && l.count.checked_add(l.off)? <= orig_len
            }
            KIND_IMAGE => {
                l.row > 0
                    && l.row % l.chans as usize == 0
                    && l.count % l.row == 0
                    && l.stride >= l.row.checked_mul(w)?
                    && (l.count / l.row).checked_mul(l.stride)?.checked_add(l.off)? <= orig_len
            }
            _ => false,
        };
        let ok = shape_ok
            && (l.flags & LAY_GDIFF == 0 || (l.kind == KIND_IMAGE && l.chans >= 3))
            && (1..=4).contains(&l.width)
            && (l.shift as u32) < l.width as u32 * 8
            && (1..=8).contains(&l.chans)
            && l.count % l.chans as usize == 0;
        ok.then_some(l)
    }
}

/// e^x for x <= 0, from `+ * /` and exponent arithmetic only, so it is
/// bit-identical everywhere (libm's exp is not required to be). Range
/// reduction to x = k ln2 + r, |r| <= ln2/2, then a degree-11 Taylor series:
/// relative error under 1e-15, far below anything a 12-bit probability sees.
fn exp_neg(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    const LN2: f64 = std::f64::consts::LN_2;
    let k = (x / LN2).round();
    let r = x - k * LN2;
    let mut t = 1.0;
    let mut sum = 1.0;
    for i in 1..12 {
        t = t * r / i as f64;
        sum += t;
    }
    sum * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

/// P(residual in the upper half | it lies in [lo, hi)), halves split at mid,
/// for a Laplace(mu, b) over integers (each integer v owns [v-0.5, v+0.5)).
fn laplace_upper(lo: f64, mid: f64, hi: f64, mu: f64, b: f64) -> f64 {
    let t1 = (lo - 0.5 - mu) / b;
    let t2 = (mid - 0.5 - mu) / b;
    let t3 = (hi - 0.5 - mu) / b;
    if t3 <= 0.0 {
        // all left of the mode: ratios of e^t, scaled by e^-t3 to stay finite
        let (a1, a2) = (exp_neg(t1 - t3), exp_neg(t2 - t3));
        (1.0 - a2) / (1.0 - a1).max(1e-300)
    } else if t1 >= 0.0 {
        let (a2, a3) = (exp_neg(t1 - t2), exp_neg(t1 - t3));
        (a2 - a3) / (1.0 - a3).max(1e-300)
    } else {
        let cdf = |t: f64| if t <= 0.0 { 0.5 * exp_neg(t) } else { 1.0 - 0.5 * exp_neg(-t) };
        let (c1, c2, c3) = (cdf(t1), cdf(t2), cdf(t3));
        (c3 - c2) / (c3 - c1).max(1e-300)
    }
}

#[inline]
fn wrap(v: i64, bits: u32) -> i64 {
    (v << (64 - bits)) >> (64 - bits)
}

// ---------------------------------------------------------------------------
// Numeric building blocks.
// ---------------------------------------------------------------------------

/// History of one signal, newest first, as a contiguous window: every value is
/// written twice, `h` apart, so `window(n)` never wraps.
struct Hist {
    buf: Vec<f64>,
    pos: usize,
    h: usize,
}

impl Hist {
    fn new(h: usize) -> Self {
        Self { buf: vec![0.0; 2 * h], pos: 0, h }
    }
    #[inline]
    fn push(&mut self, v: f64) {
        self.pos = if self.pos == 0 { self.h - 1 } else { self.pos - 1 };
        self.buf[self.pos] = v;
        self.buf[self.pos + self.h] = v;
    }
    #[inline]
    fn window(&self, n: usize) -> &[f64] {
        &self.buf[self.pos..self.pos + n]
    }
    #[inline]
    fn at(&self, k: usize) -> f64 {
        self.buf[self.pos + k]
    }
}

/// Least squares with exponential forgetting: keeps the weighted covariance of
/// the inputs and their correlation with the target, and every `interval`
/// samples re-solves the normal equations by Cholesky. This is the strongest
/// linear predictor there is for a stationary stretch of signal — it finds the
/// optimal coefficients outright, where an LMS filter only drifts toward them.
struct Ols {
    n: usize,
    interval: usize,
    lambda: f64,
    nu: f64,
    x: Vec<f64>,
    w: Vec<f64>,
    b: Vec<f64>,
    cov: Vec<f64>, // lower triangle, row-major n*n
    chol: Vec<f64>,
    since: usize,
}

impl Ols {
    fn new(n: usize, interval: usize, lambda: f64, nu: f64) -> Self {
        Self {
            n,
            interval,
            lambda,
            nu,
            x: vec![0.0; n],
            w: vec![0.0; n],
            b: vec![0.0; n],
            cov: vec![0.0; n * n],
            chol: vec![0.0; n * n],
            since: 0,
        }
    }

    #[inline]
    fn predict(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.n {
            s += self.x[i] * self.w[i];
        }
        s
    }

    fn update(&mut self, y: f64) {
        let (a, beta) = (self.lambda, 1.0 - self.lambda);
        let n = self.n;
        for i in 0..n {
            let xb = self.x[i] * beta;
            let row = &mut self.cov[i * n..i * n + i + 1];
            for j in 0..=i {
                row[j] = a * row[j] + self.x[j] * xb;
            }
            self.b[i] = a * self.b[i] + y * self.x[i] * beta;
        }
        self.since += 1;
        if self.since >= self.interval {
            self.since = 0;
            if self.factor() {
                self.solve();
            }
        }
    }

    fn factor(&mut self) -> bool {
        let n = self.n;
        for i in 0..n {
            for j in 0..i {
                self.chol[i * n + j] = self.cov[i * n + j];
            }
            self.chol[i * n + i] = self.cov[i * n + i] + self.nu;
        }
        for i in 0..n {
            for j in 0..=i {
                let mut s = self.chol[i * n + j];
                for k in 0..j {
                    s -= self.chol[i * n + k] * self.chol[j * n + k];
                }
                if i == j {
                    if s <= 1e-8 {
                        return false;
                    }
                    self.chol[i * n + i] = s.sqrt();
                } else {
                    self.chol[i * n + j] = s / self.chol[j * n + j];
                }
            }
        }
        true
    }

    fn solve(&mut self) {
        let n = self.n;
        // L z = b, then L^T w = z
        for i in 0..n {
            let mut s = self.b[i];
            for k in 0..i {
                s -= self.chol[i * n + k] * self.w[k];
            }
            self.w[i] = s / self.chol[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = self.w[i];
            for k in i + 1..n {
                s -= self.chol[k * n + i] * self.w[k];
            }
            self.w[i] = s / self.chol[i * n + i];
        }
    }
}

/// LMS with a per-weight RMS-normalised step (RMSprop). Plain LMS steps in
/// proportion to the signal's scale, so one rate cannot suit both a whisper and
/// a crescendo; normalising each weight's gradient by its running magnitude
/// makes the rate scale-free.
struct Lms {
    w: Vec<f64>,
    eg: Vec<f64>,
    rate: f64,
    pred: f64,
}

const LMS_RHO: f64 = 0.995;
const LMS_EPS: f64 = 1e-3;

impl Lms {
    fn new(n: usize, rate: f64) -> Self {
        Self { w: vec![0.0; n], eg: vec![0.0; n], rate, pred: 0.0 }
    }
    #[inline]
    fn predict(&mut self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..self.w.len() {
            s += self.w[i] * x[i];
        }
        self.pred = s;
        s
    }
    #[inline]
    fn update(&mut self, x: &[f64], actual: f64) {
        let err = actual - self.pred;
        for i in 0..self.w.len() {
            let g = err * x[i];
            self.eg[i] = LMS_RHO * self.eg[i] + (1.0 - LMS_RHO) * g * g;
            self.w[i] += self.rate * g / (self.eg[i] + LMS_EPS).sqrt();
        }
    }
}

// ---------------------------------------------------------------------------
// Audio model.
// ---------------------------------------------------------------------------

/// Number of predictors feeding per-bit contexts. Index 0 is the main
/// prediction whose residual is coded.
pub const NP: usize = 12; // audio
/// Most predictors any model supplies (images).
pub const NPMAX: usize = 27;
/// Direct maps per predictor: two residual resolutions, and one for the
/// predictor corrected by its own last error.
const MAPS_PER: usize = 4;
/// Parametric inputs: Laplace distributions around chosen predictions.
const NPAR: usize = 4;
/// Most value-domain hashed contexts a model supplies (images).
pub const NVMAX: usize = 16;
const VTAB_BITS: u32 = 23;
const HMAP_BITS: u32 = 23;
pub const FRONT_IN: usize = NPAR + NPMAX * MAPS_PER + 2 * NVMAX;

const HIST: usize = 2048;
const OLS_A_OWN: usize = 64;
const OLS_A_OTHER: usize = 32;
const CASC1: usize = 256;

/// Lags for the sparse OLS: every sample for the first 16, then steps that
/// double every 8 taps, out to 1024 back. Pitch periods live out there, and a
/// dense filter that long would cost a fortune to solve.
fn sparse_lags() -> Vec<usize> {
    let mut v: Vec<usize> = (0..16).collect();
    let mut step = 2;
    let mut at = 16;
    while at < 1024 {
        for _ in 0..8 {
            v.push(at);
            at += step;
        }
        step *= 2;
    }
    v
}
const CASC2: usize = 16;
const LMSD_OWN: usize = 256;
const LMSD_OTHER: usize = 32;

struct Channel {
    x: Hist,     // reconstructed samples
    e: Hist,     // OLS-A residuals, the cascade's input
    ols_a: Ols,  // long, cross-channel
    ols_b: Ols,  // medium, own channel only
    ols_c: Ols,  // short and fast-forgetting
    ols_s: Ols,  // sparse lags reaching far back
    casc1: Lms,  // cascade stages on OLS-A's residual
    casc2: Lms,
    lmsd: Lms,   // direct LMS on samples (own + other channels)
    lms_in: Vec<f64>,
    // per-predictor outputs for the sample being coded, and last errors
    p: [f64; NP],
    last_err: [f64; NP],
    // pieces of the cascade, kept for its update
    pa: f64,
    c1: f64,
    // magnitude statistics of the main residual
    err_fast: f64,
    err_slow: f64,
    last_res: i64,
}

pub struct AudioModel {
    lags: Vec<usize>,
    chans: usize,
    bits: u32,
    ch: Vec<Channel>,
    cur: usize, // channel of the next sample
}

impl AudioModel {
    pub fn new(chans: usize, bits: u32) -> Self {
        let other = if chans > 1 { OLS_A_OTHER / (chans - 1).max(1) } else { 0 };
        let na = OLS_A_OWN + other * (chans - 1);
        let nd = LMSD_OWN + if chans > 1 { LMSD_OTHER } else { 0 };
        let ch = (0..chans)
            .map(|_| Channel {
                x: Hist::new(HIST),
                e: Hist::new(HIST),
                ols_a: Ols::new(na, 16, 0.998, 0.001),
                ols_b: Ols::new(16, 4, 0.999, 0.001),
                ols_c: Ols::new(8, 1, 0.99, 0.001),
                ols_s: Ols::new(sparse_lags().len() + if chans > 1 { 8 } else { 0 }, 16, 0.999, 0.001),
                casc1: Lms::new(CASC1, 0.0006),
                casc2: Lms::new(CASC2, 0.002),
                lmsd: Lms::new(nd, 0.0002),
                lms_in: vec![0.0; nd],
                p: [0.0; NP],
                last_err: [0.0; NP],
                pa: 0.0,
                c1: 0.0,
                err_fast: 0.0,
                err_slow: 0.0,
                last_res: 0,
            })
            .collect();
        Self { lags: sparse_lags(), chans, bits, ch, cur: 0 }
    }

    fn other_per(&self) -> usize {
        if self.chans > 1 {
            OLS_A_OTHER / (self.chans - 1)
        } else {
            0
        }
    }

    /// Run every predictor for the next sample; returns the main prediction.
    fn predict(&mut self) {
        let c = self.cur;
        let other = self.other_per();
        // gather cross-channel inputs before borrowing this channel mutably
        let mut xa: Vec<f64> = Vec::with_capacity(64);
        xa.extend_from_slice(self.ch[c].x.window(OLS_A_OWN));
        let mut xd: Vec<f64> = Vec::with_capacity(LMSD_OWN + LMSD_OTHER);
        xd.extend_from_slice(self.ch[c].x.window(LMSD_OWN));
        let mut cross = 0.0;
        for o in 0..self.chans {
            if o == c {
                continue;
            }
            // channels before this one already hold this frame's sample
            xa.extend_from_slice(self.ch[o].x.window(other));
            if xd.len() < LMSD_OWN + LMSD_OTHER {
                xd.extend_from_slice(self.ch[o].x.window(LMSD_OTHER));
            }
            if o < c {
                cross = self.ch[o].x.at(0);
            }
        }
        let mut xs: Vec<f64> = self.lags.iter().map(|&l| self.ch[c].x.at(l)).collect();
        if self.chans > 1 {
            xs.extend_from_slice(self.ch[(c + 1) % self.chans].x.window(8));
        }
        let lim = ((1i64 << (self.bits - 1)) - 1) as f64;
        let ch = &mut self.ch[c];
        ch.ols_a.x.copy_from_slice(&xa);
        ch.ols_b.x.copy_from_slice(ch.x.window(16));
        ch.ols_c.x.copy_from_slice(ch.x.window(8));
        ch.ols_s.x.copy_from_slice(&xs);
        ch.lms_in.copy_from_slice(&xd);

        let pl = ch.ols_a.predict();
        let ps = ch.ols_s.predict();
        // the dense and the sparse fit see different structure; their mean
        // beats either as the cascade's base
        let pa = (pl + ps) * 0.5;
        let c1 = ch.casc1.predict(ch.e.window(CASC1));
        let c2 = ch.casc2.predict(ch.e.window(CASC2));
        ch.pa = pa;
        ch.c1 = c1;
        let main = pa + c1 + c2;
        let (x1, x2, x3) = (ch.x.at(0), ch.x.at(1), ch.x.at(2));
        let p = [
            main,
            pa,
            pa + c1,
            ch.ols_b.predict(),
            ch.ols_c.predict(),
            ch.lmsd.predict(&ch.lms_in),
            x1,
            2.0 * x1 - x2,
            3.0 * x1 - 3.0 * x2 + x3,
            if c > 0 { cross } else { x1 + (x1 - x2) * 0.5 },
            main + ch.last_err[0] * 0.5,
            ps,
        ];
        for i in 0..NP {
            ch.p[i] = p[i].clamp(-lim - 1.0, lim);
        }
    }

    fn prep(&mut self, o: &mut Prep) {
        self.predict();
        let c = self.cur;
        let ch = &self.ch[c];
        for i in 0..NP {
            o.p[i] = ch.p[i];
            o.pe[i] = ch.p[i] + ch.last_err[i];
        }
        let e = ch.err_fast.max(1.0) as u64;
        o.errlog = (64 - e.leading_zeros()).min(15);
        o.scale = [ch.err_fast, ch.err_slow];
        let es = (64 - (ch.err_slow.max(1.0) as u64).leading_zeros()).min(31);
        let d = |i: usize| (ch.p[i] - ch.p[0]).round() as i64;
        let el = o.errlog;
        o.sctx = [
            el,
            el << 5 | es,
            q(d(1)) << 6 | q(d(3)),
            q(ch.last_res) << 4 | el,
            q(d(5)) << 6 | q(d(4)),
            (c as u32) << 8 | q(d(9)) << 4 | el,
            q(d(2)) << 4 | el,
        ];
    }

    fn update(&mut self, x: i64, res: i64) {
        let c = self.cur;
        let ch = &mut self.ch[c];
        let xf = x as f64;
        ch.ols_a.update(xf);
        ch.ols_b.update(xf);
        ch.ols_c.update(xf);
        ch.ols_s.update(xf);
        let ea = xf - ch.pa;
        ch.casc1.update(ch.e.window(CASC1), ea);
        ch.casc2.update(ch.e.window(CASC2), ea - ch.c1);
        ch.lmsd.update(&ch.lms_in, xf);
        for i in 0..NP {
            ch.last_err[i] = xf - ch.p[i];
        }
        ch.e.push(ea);
        ch.x.push(xf);
        let a = res.unsigned_abs() as f64;
        ch.err_fast = ch.err_fast * 0.8 + a * 0.2;
        ch.err_slow = ch.err_slow * 0.98 + a * 0.02;
        ch.last_res = res;
        self.cur = (c + 1) % self.chans;
    }
}

// ---------------------------------------------------------------------------
// Image model.
//
// Components are coded in raster order, pixel by pixel, so when the green of a
// pixel is coded its red is already known. Every predictor here is a classic
// one — LOCO-I's median edge detector, CALIC's gradient-adjusted predictor,
// planar and directional extrapolations, and cross-component versions that
// borrow the previous component's local slope — plus a least-squares fit over
// the whole causal neighbourhood. The main prediction blends them, each
// weighted by the inverse square of the error it made at the neighbouring
// pixels, so near an edge the predictor that respects the edge takes over.
// ---------------------------------------------------------------------------

const IMG_ROWS: usize = 8; // rows of history kept, ring-indexed
/// Predictors the image model supplies.
pub const NPI: usize = 27;
const IMG_X_N: usize = 15; // cross-component OLS inputs, at most

/// Same-component taps (dy rows up, dx columns right) for the spatial OLS
/// fits: a wide one, and narrower ones leaning north, west, or short and fast.
const TAPS_WIDE: [(u8, i8); 32] = [
    (0, -6), (0, -5), (0, -4), (0, -3), (0, -2), (0, -1),
    (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
    (3, -2), (3, -1), (3, 0), (3, 1), (3, 2),
    (4, -1), (4, 0), (4, 1), (5, 0), (6, 0),
];
const TAPS_NEAR: [(u8, i8); 12] =
    [(0, -3), (0, -2), (0, -1), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -1), (2, 0), (2, 1), (3, 0)];
const TAPS_NORTH: [(u8, i8); 15] = [
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (5, 0),
];
const TAPS_WEST: [(u8, i8); 14] = [
    (0, -4), (0, -3), (0, -2), (0, -1), (1, -3), (1, -2), (1, -1), (1, 0), (2, -2), (2, -1), (2, 0), (3, -1), (3, 0), (4, 0),
];
const TAPS_FAST: [(u8, i8); 6] = [(0, -3), (0, -2), (0, -1), (3, 0), (2, 0), (1, 0)];

pub struct ImageModel {
    palette: bool,
    run: u32, // how far the current row's run of equal pixels has gone
    comps: usize,
    rowlen: usize,     // samples per row
    px: Vec<i32>,      // reconstructed samples, IMG_ROWS rows
    err: Vec<[i16; NPI]>, // signed error of each predictor, same ring
    s: usize,          // index of the sample being predicted
    // per component: cross-component fit, then the spatial ones
    ols_x: Vec<Ols>,
    ols_sp: Vec<[Ols; 5]>,
    p: [f64; NPI],
    bias: Vec<(i32, i32)>, // per texture context: error sum, count
    bcx: usize,
    blend: f64,
    ols_err: Vec<f64>, // slow average |main residual| per component
    // colour cache: (component, the components already known at this pixel)
    // -> the last two values this component took there
    cache: Vec<[i32; 2]>,
    ckey: usize,
    // cross-colour caches: the previous component's exact value with this
    // component's colour difference at a neighbour -> what this one was
    xcache: Vec<i32>,
    xkey: [usize; NXC],
}

const NXC: usize = 3;



const CACHE_BITS: u32 = 20;

#[inline]
fn lg2i(v: u64) -> u32 {
    (64 - v.leading_zeros()).min(31)
}

#[inline]
fn med(w: i32, n: i32, nw: i32) -> i32 {
    let (lo, hi) = if w < n { (w, n) } else { (n, w) };
    if nw >= hi {
        lo
    } else if nw <= lo {
        hi
    } else {
        w + n - nw
    }
}

impl ImageModel {
    fn new(width: usize, comps: usize, _bits: u32, palette: bool) -> Self {
        let rowlen = width * comps;
        let nx = |c: usize| 7 + if c > 0 { 5 } else { 0 } + if c > 1 { 3 } else { 0 };
        Self {
            palette,
            run: 0,
            comps,
            rowlen,
            px: vec![0; IMG_ROWS * rowlen],
            err: vec![[0; NPI]; IMG_ROWS * rowlen],
            s: 0,
            ols_x: (0..comps).map(|c| Ols::new(nx(c), 1, 0.998, 0.001)).collect(),
            ols_sp: (0..comps)
                .map(|_| {
                    [
                        Ols::new(TAPS_WIDE.len(), 1, 0.98, 0.001),
                        Ols::new(TAPS_NEAR.len(), 1, 0.87, 0.001),
                        Ols::new(TAPS_NORTH.len(), 1, 0.9, 0.001),
                        Ols::new(TAPS_WEST.len(), 1, 0.9, 0.001),
                        Ols::new(TAPS_FAST.len() + 2, 1, 0.7, 0.001),
                    ]
                })
                .collect(),
            p: [0.0; NPI],
            bias: vec![(0, 0); 3 * 3 * 3 * 16 * 4],
            bcx: 0,
            blend: 0.0,
            ols_err: vec![4.0; comps],
            cache: vec![[i32::MIN; 2]; 1 << CACHE_BITS],
            ckey: 0,
            xcache: vec![i32::MIN; 1 << CACHE_BITS],
            xkey: [0; NXC],

        }
    }

    #[inline]
    fn slot(&self, dy: usize, x: usize, c: usize) -> usize {
        let y = self.s / self.rowlen;
        ((y - dy) % IMG_ROWS) * self.rowlen + x * self.comps + c
    }

    /// Component `c` at (dy rows up, dx columns right) of the current pixel,
    /// if it has been coded.
    #[inline]
    fn get(&self, dy: usize, dx: isize, c: usize) -> Option<i32> {
        let y = self.s / self.rowlen;
        let x = (self.s % self.rowlen) / self.comps;
        let xx = x as isize + dx;
        let wd = (self.rowlen / self.comps) as isize;
        (dy <= y && xx >= 0 && xx < wd && (dy > 0 || xx < x as isize)).then(|| self.px[self.slot(dy, xx as usize, c)])
    }

    fn prep(&mut self, o: &mut Prep) {
        let y = self.s / self.rowlen;
        let i = self.s % self.rowlen;
        let (x, c) = (i / self.comps, i % self.comps);
        let wd = self.rowlen / self.comps;
        // missing neighbours (first rows and columns) borrow from ones that exist
        let w0 = self.get(0, -1, c);
        let fb = self.get(1, 0, c).or(w0).unwrap_or(0);
        // the causal neighbourhood, gathered once: rows 0..=6 up, columns -6..=6
        let mut grid = [[0i32; 13]; 7];
        for (dy, row) in grid.iter_mut().enumerate() {
            for (k, v) in row.iter_mut().enumerate() {
                *v = self.get(dy, k as isize - 6, c).unwrap_or(fb);
            }
        }
        let g = |dy: usize, dx: isize| grid[dy][(dx + 6) as usize];
        let (w, n, nw, ne) = (g(0, -1), g(1, 0), g(1, -1), g(1, 1));
        let (ww, nn, nne) = (g(0, -2), g(2, 0), g(2, 1));
        let (www, wwww, wwwww) = (g(0, -3), g(0, -4), g(0, -5));
        let nnn = g(3, 0);
        let (nnne, nnnne, nnee, nnww, nnnwww) = (g(3, 1), g(4, 1), g(2, 2), g(2, -2), g(3, -3));
        // CALIC's gradient-adjusted prediction
        let dh = (w - ww).abs() + (n - nw).abs() + (n - ne).abs();
        let dv = (w - nw).abs() + (n - nn).abs() + (ne - nne).abs();
        let gap = if dv - dh > 80 {
            w
        } else if dh - dv > 80 {
            n
        } else {
            let t = (w + n) / 2 + (ne - nw) / 4;
            match dv - dh {
                d if d > 32 => (t + w) / 2,
                d if d > 8 => (3 * t + w) / 4,
                d if d < -32 => (t + n) / 2,
                d if d < -8 => (3 * t + n) / 4,
                _ => t,
            }
        };
        let m = med(w, n, nw);
        // previous components of this pixel (or the last ones of the pixel to
        // the left), for the cross-component and fast fits
        let prev = |k: usize| -> i32 {
            if c >= k {
                self.px[self.slot(0, x, c - k)]
            } else if x > 0 {
                self.px[self.slot(0, x - 1, c + self.comps - k)]
            } else {
                fb
            }
        };
        let (prev1, prev2) = (prev(1), if self.comps >= 2 { prev(2) } else { prev(1) });
        let mut cross = [2 * w - ww, 2 * n - nn, ne, n + ne - nne];
        let mut xin = [0f64; IMG_X_N];
        let base = [n, w, nw, ne, ww, nn, nne];
        for k in 0..7 {
            xin[k] = base[k] as f64;
        }
        let mut nin = 7;
        if c > 0 {
            let bg = |dy: usize, dx: isize| self.get(dy, dx, c - 1).unwrap_or(fb);
            let (bw, bn, bnw, bne) = (bg(0, -1), bg(1, 0), bg(1, -1), bg(1, 1));
            let cb = self.px[self.slot(0, x, c - 1)];
            cross = [w + cb - bw, n + cb - bn, m + cb - med(bw, bn, bnw), w + n - nw + cb - (bw + bn - bnw)];
            for v in [cb, bw, bn, bnw, bne] {
                xin[nin] = v as f64;
                nin += 1;
            }
            if c > 1 {
                let c0 = self.px[self.slot(0, x, 0)];
                for v in [c0, self.get(0, -1, 0).unwrap_or(fb), self.get(1, 0, 0).unwrap_or(fb)] {
                    xin[nin] = v as f64;
                    nin += 1;
                }
            }
        }
        let ox = &mut self.ols_x[c];
        ox.x[..nin].copy_from_slice(&xin[..nin]);
        let p_x = ox.predict();
        let mut p_sp = [0f64; 5];
        {
            let taps: [&[(u8, i8)]; 4] = [&TAPS_WIDE, &TAPS_NEAR, &TAPS_NORTH, &TAPS_WEST];
            let mut xs = [0f64; 32];
            for (k, t) in taps.iter().enumerate() {
                for (j, &(dy, dx)) in t.iter().enumerate() {
                    xs[j] = g(dy as usize, dx as isize) as f64;
                }
                let o2 = &mut self.ols_sp[c][k];
                o2.x.copy_from_slice(&xs[..t.len()]);
                p_sp[k] = o2.predict();
            }
            for (j, &(dy, dx)) in TAPS_FAST.iter().enumerate() {
                xs[j] = g(dy as usize, dx as isize) as f64;
            }
            xs[6] = prev1 as f64;
            xs[7] = prev2 as f64;
            let o2 = &mut self.ols_sp[c][4];
            o2.x.copy_from_slice(&xs[..8]);
            p_sp[4] = o2.predict();
        }
        // colour cache key: earlier components of this pixel exactly, or for the
        // first component the whole colour of the pixel to the left
        let mut key: u64 = c as u64 + 1;
        if c > 0 {
            for k in 0..c {
                key = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (self.px[self.slot(0, x, k)] as u32 as u64);
            }
        } else {
            for k in 0..self.comps {
                key = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (self.get(0, -1, k).unwrap_or(i32::MIN) as u32 as u64);
            }
        }
        key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        self.ckey = (key >> (64 - CACHE_BITS)) as usize;
        let [c1, c2] = self.cache[self.ckey];
        // an empty entry has nothing to say; stand in plausible predictions
        let (c1, c2) = (if c1 == i32::MIN { m } else { c1 }, if c2 == i32::MIN { gap } else { c2 });
        // cross-colour caches; the first component has no earlier one, so it
        // keys on its own neighbours' differences instead
        let xv = {
            let (cb, dw, dn, c0) = if c > 0 {
                let bw = self.get(0, -1, c - 1).unwrap_or(fb);
                let bn = self.get(1, 0, c - 1).unwrap_or(fb);
                (self.px[self.slot(0, x, c - 1)], w - bw, n - bn, if c > 1 { self.px[self.slot(0, x, 0)] } else { 0 })
            } else {
                (w, w - nw, n - nw, ne - n)
            };
            let keys = [(cb, dw, 0), (cb, dn, 0), (cb, dw, c0 ^ 0x5555)];
            let mut out = [m; NXC];
            for (j, &(a, b, e)) in keys.iter().enumerate() {
                let mut hh = ((c as u64 + 1) << 40) ^ (j as u64) << 50;
                for v in [a, b, e] {
                    hh = (hh ^ v as u32 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                }
                self.xkey[j] = (hh >> (64 - CACHE_BITS)) as usize;
                let e = self.xcache[self.xkey[j]];
                if e != i32::MIN {
                    out[j] = e;
                }
            }
            out
        };
        let f = |v: i32| v as f64;
        let cand: [f64; NPI - 1] = [
            p_x,
            p_sp[0],
            p_sp[1],
            p_sp[2],
            p_sp[3],
            p_sp[4],
            f(m),
            f(gap),
            f(w),
            f(n),
            f(w + n - nw),
            f(w + ne) / 2.0,
            f(cross[0]),
            f(cross[1]),
            f(cross[2]),
            f(cross[3]),
            f(ne),
            f(6 * n - 4 * nn + nnn + 4 * ne - 6 * nne + 4 * nnne - nnnne) / 4.0,
            f(15 * w - 20 * ww + 15 * www - 6 * wwww + wwwww) / 5.0,
            f((2 * ne - nnee) + (2 * nw - nnww) - (2 * n - nn)),
            f((6 * w - 4 * ww + www) + (6 * n - 4 * nn + nnn) - (6 * nw - 4 * nnww + nnnwww)) / 3.0,
            f(c1),
            f(c2),
            f(xv[0]),
            f(xv[1]),
            f(xv[2]),
        ];
        // weight each candidate by its recent accuracy around this pixel
        let mut score = [1u64; NPI - 1];
        let mut eadd = |k: usize, wgt: u64| {
            for j in 0..NPI - 1 {
                score[j] += self.err[k][j + 1].unsigned_abs() as u64 * wgt;
            }
        };
        if x > 0 {
            eadd(self.slot(0, x - 1, c), 2);
        }
        if y > 0 {
            eadd(self.slot(1, x, c), 2);
            if x > 0 {
                eadd(self.slot(1, x - 1, c), 1);
            }
            if x + 1 < wd {
                eadd(self.slot(1, x + 1, c), 1);
            }
        }
        let (mut num, mut den) = (0f64, 0f64);
        for j in 0..NPI - 1 {
            let wgt = 1.0 / (score[j] as f64 * score[j] as f64);
            num += wgt * cand[j];
            den += wgt;
        }
        let blend = num / den;
        self.blend = blend;
        // bias cancellation per texture context
        let sg = |d: i32| (d.signum() + 1) as usize;
        let act = (64 - score.iter().min().unwrap().leading_zeros()).min(15) as usize;
        self.bcx = ((((sg(n - nw) * 3 + sg(w - nw)) * 3 + sg(ne - n)) * 16 + act) * 4) + c.min(3);
        let (bs, bn) = self.bias[self.bcx];
        let main = if bn > 0 { blend + bs as f64 / bn as f64 } else { blend };
        self.p[0] = main;
        self.p[1..].copy_from_slice(&cand);
        // neighbours' errors: the corrected predictions and the contexts
        let ew = if x > 0 { self.err[self.slot(0, x - 1, c)] } else { [0; NPI] };
        let en = if y > 0 { self.err[self.slot(1, x, c)] } else { [0; NPI] };
        for j in 0..NPI {
            o.p[j] = self.p[j];
            o.pe[j] = self.p[j] + (ew[j] as f64 + en[j] as f64) * 0.5;
        }
        let ecur = |cc: usize| if c > cc { self.err[self.slot(0, x, cc)][0] as i64 } else { 0 };
        let el = 64 - ((ew[0].unsigned_abs() as u64 + en[0].unsigned_abs() as u64) * 2 + 1).leading_zeros();
        o.errlog = el.min(15);
        o.scale = [(ew[0].unsigned_abs() as f64 + en[0].unsigned_abs() as f64) * 0.5, self.ols_err[c]];
        let d = |j: usize| (self.p[j] - main).round() as i64;
        let cc = c.min(3) as u32;
        {
            let (cb, bw, bn) = if c > 0 {
                (self.px[self.slot(0, x, c - 1)], self.get(0, -1, c - 1).unwrap_or(fb), self.get(1, 0, c - 1).unwrap_or(fb))
            } else {
                (prev1, prev1, prev1)
            };
            let c0v = if c > 1 { self.px[self.slot(0, x, 0)] } else { 0 };
            let hv = |tag: u32, a: i32, b: i32, e: i32| -> u32 {
                (tag << 28 | cc << 26)
                    ^ (a as u32).wrapping_mul(0x9e37_79b1)
                    ^ (b as u32).wrapping_mul(0x85eb_ca6b)
                    ^ (e as u32).wrapping_mul(0xc2b2_ae35)
            };
            let r = |v: f64| v.round() as i32;
            // the byte k before the same component of a neighbour, in file
            // order: for the pixel above, "its p1"
            let before = |dy: usize, dx: isize, k: usize| -> i32 {
                // row-linear sample position, k samples back, in any layout
                let lin = (x as isize + dx) * self.comps as isize + c as isize - k as isize;
                if lin < 0 {
                    return fb;
                }
                let (px_, cc2) = (lin as usize / self.comps, lin as usize % self.comps);
                let rel = px_ as isize - x as isize;
                self.get(dy, rel, cc2)
                    .or_else(|| (dy == 0 && rel == 0 && cc2 < c).then(|| self.px[self.slot(0, x, cc2)]))
                    .unwrap_or(fb)
            };
            let (p1, p2) = (prev1, prev2);
            let (wp1, wp2) = (before(0, -1, 1), before(0, -1, 2));
            let (np1, np2) = (before(1, 0, 1), before(1, 0, 2));
            let plane = (w + n - nw) >> 1;
            if self.palette {
                // indices: only equality means anything, so every context is
                // an exact pattern of neighbours
                let nne = g(2, 1);
                let nee = g(1, 2);
                let www = g(0, -3);
                let run = self.run.min(31) as i32;
                o.vctx = [
                    hv(1, w, n, 0),
                    hv(2, w, n, nw << 8 | ne),
                    hv(3, n, ne, nne),
                    hv(4, w, ww, www),
                    hv(5, n, nn, 0),
                    hv(6, w, nw, nn),
                    hv(7, ne, nee, n),
                    hv(8, w, run, 0),
                    hv(9, n, run, (w == n) as i32),
                    hv(10, w, n, ne << 8 | nne),
                    hv(11, nw, n, ne),
                    hv(12, w, ww, n << 8 | nn),
                    hv(13, x as i32, n, 0),
                    hv(14, w, (n == ne) as i32 | ((nw == n) as i32) << 1 | ((ww == w) as i32) << 2, ne),
                    hv(15, n << 8 | nn, ne << 8 | nne, 0),
                    hv(16, w << 8 | nw, n, ne << 8 | nee),
                ];
            } else {
            o.vctx = [
                    hv(1, w, n, 0),
                    hv(2, w, nw, 0),
                    hv(3, n, ne, 0),
                    hv(4, cb, w - bw, 0),
                    hv(5, cb, n - bn, 0),
                    hv(6, cb, c0v, 0),
                    hv(7, w + n - nw, 0, 0),
                    hv(8, m, gap, 0),
                    hv(9, cb + w - bw, cb + n - bn, 0),
                    hv(10, r(main), 0, 0),
                    hv(11, w >> 2, n >> 2, (nw >> 2) << 8 | (ne >> 2) & 255),
                    hv(12, r(main) >> 1, cb, 0),
                    // paq8px's strongest: the previous bytes against a neighbour's
                    hv(13, n, p1 - np1, p2 - np2),
                    hv(14, w, p1 - wp1, p2 - wp2),
                    hv(15, plane, p1, p2),
                    hv(16, p1, p2, 0),
                ];
            }
            o.nv = NVMAX;
            for j in 0..NPI {
                let e = ew[j].unsigned_abs() as u64 + en[j].unsigned_abs() as u64;
                o.pctx[j] = (lg2i(e).min(15) as u8) << 2 | cc as u8;
            }
            // which candidate has been most accurate around here
            let best = (0..NPI - 1).min_by_key(|&j| score[j]).unwrap_or(0) as u32;
            o.msel = [1 + (cc << 3), best << 2 | cc, o.errlog << 2 | cc, (lg2i(score[best as usize]) << 2) | cc];
        }
        o.sctx = [
            cc << 4 | o.errlog,
            (self.bcx as u32) >> 2 | cc << 12,
            q(d(1)) << 7 | q(d(2)) << 1 | cc << 14,
            cc << 12 | q(ew[0] as i64) << 6 | q(en[0] as i64),
            cc << 12 | q(ecur(0)) << 6 | q(ecur(1)),
            q(d(22)) << 6 | q(d(23)) | cc << 12,
            cc << 10 | o.errlog << 6 | q((w - n) as i64),
        ];
    }

    fn update(&mut self, x: i64, _r: i64) {
        let i = self.s % self.rowlen;
        let (px, c) = (i / self.comps, i % self.comps);
        if self.palette {
            let w = if px > 0 { self.px[self.slot(0, px - 1, c)] } else { -1 };
            self.run = if w == x as i32 { self.run + 1 } else { 0 };
        }
        let k = self.slot(0, px, c);
        self.px[k] = x as i32;
        let xf = x as f64;
        for j in 0..NPI {
            self.err[k][j] = (xf - self.p[j]).round().clamp(-32768.0, 32767.0) as i16;
        }
        self.ols_x[c].update(xf);
        for o in self.ols_sp[c].iter_mut() {
            o.update(xf);
        }
        // learn the bias of the uncorrected blend; learning the corrected one
        // would settle at half the true offset
        let e = &mut self.bias[self.bcx];
        e.0 += (xf - self.blend).round() as i32;
        e.1 += 1;
        if e.1 >= 256 {
            e.0 /= 2;
            e.1 /= 2;
        }
        self.ols_err[c] = self.ols_err[c] * 0.99 + (xf - self.p[0]).abs() * 0.01;
        let e = &mut self.cache[self.ckey];
        if e[0] != x as i32 {
            *e = [x as i32, e[0]];
        }
        for j in 0..NXC {
            self.xcache[self.xkey[j]] = x as i32;
        }
        self.s += 1;
    }
}

// ---------------------------------------------------------------------------
// Front: drives a model across the coded byte stream.
// ---------------------------------------------------------------------------

/// Bits of residual resolution kept by each map's context.
const MAP_BITS: [u32; MAPS_PER] = [8, 12, 4, 5];
const MAP_LIMIT: [u32; MAPS_PER] = [127, 255, 63, 63];

pub struct SampleFront {
    pub lay: Layout,
    model: Model,
    prep: Prep,
    bits: u32,  // significant bits of a sample
    cbits: u32, // coded bits per residual (whole bytes)
    cb: usize,
    start: usize, // coded-stream offsets of the sample region
    end: usize,
    pub active: bool,
    s: usize,     // index of the sample being coded
    k: usize,     // byte within it
    known: u64,   // bytes of this residual already coded
    pred: i64,
    np: usize,    // predictors this model supplies
    d: [i64; NPMAX], // each predictor relative to the main one, residual domain
    d1: [i64; NPMAX],
    maps: Vec<u32>,
    map_off: [usize; NPMAX * MAPS_PER],
    idx: [usize; NPMAX * MAPS_PER],
    pub out: Vec<i32>,
    vtab: Vec<u32>,
    vhist: Vec<u8>,
    vsm: Vec<u32>,
    stt: &'static StateTab,
    vidx: [usize; NVMAX],
    partial_bits: u32,
    v0: u32,
    pos: u32,
}

#[inline]
fn clampn(v: i64, n: u32) -> u64 {
    let h = 1i64 << (n - 1);
    (v.clamp(-h, h - 1) + h) as u64
}

/// Signed log-magnitude bucket: sign and bit length.
#[inline]
fn q(v: i64) -> u32 {
    let m = 64 - v.unsigned_abs().leading_zeros();
    (m << 1) | (v < 0) as u32
}

const NSCTX: usize = 7;

/// What a model hands the front-end before each sample.
struct Prep {
    p: [f64; NPMAX],  // every predictor's guess; p[0] is the main one
    pe: [f64; NPMAX], // each corrected by its recent error
    errlog: u32,   // how surprising this neighbourhood has been, log scale
    scale: [f64; 2], // expected |main residual|: fast- and slow-tracking
    sctx: [u32; NSCTX], // model-specific byte contexts
    // value-domain contexts, hashed per bit with the bits of the coded value
    // known so far: only meaningful when the value itself is coded
    vctx: [u32; NVMAX],
    nv: usize,
    // per predictor: a small context for its maps (its recent error, the
    // component); when nonzero the maps are hashed with it
    pctx: [u8; NPMAX],
    // mixer selectors for models that want their own (0 = none)
    msel: [u32; 4],
}

enum Model {
    Audio(AudioModel),
    Image(ImageModel),
}

impl Model {
    fn prep(&mut self, o: &mut Prep) {
        match self {
            Model::Audio(m) => m.prep(o),
            Model::Image(m) => m.prep(o),
        }
    }
    fn update(&mut self, x: i64, r: i64) {
        match self {
            Model::Audio(m) => m.update(x, r),
            Model::Image(m) => m.update(x, r),
        }
    }
}

impl SampleFront {
    /// A front-end for `lay`, whose first coded byte is at `start`; `buf` is
    /// the coded stream so far.
    pub fn new(lay: Layout, start: usize, buf: &[u8]) -> Self {
        let bits = lay.bits();
        let cb = lay.code_bytes();
        let np = if lay.kind == KIND_IMAGE { NPI } else { NP };
        let mut map_off = [0usize; NPMAX * MAPS_PER];
        let mut total = 0;
        for i in 0..np {
            for j in 0..MAPS_PER {
                map_off[i * MAPS_PER + j] = total;
                total += 1 << (MAP_BITS[j] + 5);
            }
        }
        let model = match lay.kind {
            KIND_IMAGE => Model::Image(ImageModel::new(lay.row / lay.chans as usize, lay.chans as usize, bits, lay.flags & LAY_PALETTE != 0)),
            _ => Model::Audio(AudioModel::new(lay.chans as usize, bits)),
        };
        let mut f = Self {
            lay,
            model,
            prep: Prep { p: [0.0; NPMAX], pe: [0.0; NPMAX], errlog: 0, scale: [1.0; 2], sctx: [0; NSCTX], vctx: [0; NVMAX], nv: 0, msel: [0; 4], pctx: [0; NPMAX] },
            bits,
            cbits: 8 * cb as u32,
            cb,
            start,
            end: start + lay.count * cb,
            active: false,
            s: 0,
            k: 0,
            known: 0,
            pred: 0,
            np,
            d: [0; NPMAX],
            d1: [0; NPMAX],
            // images hash their maps with a per-predictor context
            maps: vec![(1 << 21) << 10; if lay.kind == KIND_IMAGE { 1 << HMAP_BITS } else { total }],
            map_off,
            idx: [0; NPMAX * MAPS_PER],
            out: Vec::new(),
            vtab: if lay.kind == KIND_IMAGE { vec![(1 << 21) << 10; 1 << VTAB_BITS] } else { Vec::new() },
            vhist: if lay.kind == KIND_IMAGE { vec![0; 1 << VTAB_BITS] } else { Vec::new() },
            vsm: (0..NVMAX).flat_map(|_| sm_init(state_tab())).collect(),
            stt: state_tab(),
            vidx: [0; NVMAX],
            partial_bits: 0,
            v0: 0,
            pos: 0,
        };
        if lay.count > 0 {
            f.prepare();
        }
        f.locate(start, buf);
        f
    }

    /// Every sample has been coded.
    pub fn finished(&self) -> bool {
        self.s >= self.lay.count
    }

    fn prepare(&mut self) {
        self.model.prep(&mut self.prep);
        let lim = ((1i64 << (self.bits - 1)) - 1) as f64;
        let o = &mut self.prep;
        for i in 0..self.np {
            o.p[i] = o.p[i].clamp(-lim - 1.0, lim);
        }
        self.pred = o.p[0].round() as i64;
        // Images are coded as values, not residuals: the predictors still
        // steer every bit through their maps, and value-domain contexts —
        // "the red here was 140" — keep their meaning, which a residual
        // relative to a moving prediction blurs.
        if matches!(self.model, Model::Image(_)) {
            self.pred = 0;
        }
        for i in 0..self.np {
            self.d[i] = o.p[i].round() as i64 - self.pred;
            self.d1[i] = o.pe[i].clamp(-lim - 1.0, lim).round() as i64 - self.pred;
        }
    }

    /// The coded bytes for sample `x` (encoder side). Must be called while the
    /// front is positioned at the start of that sample.
    pub fn code(&self, x: i64) -> [u8; 4] {
        let r = wrap(x - self.pred, self.bits);
        let u = (r + (1i64 << (self.cbits - 1))) as u64;
        let mut b = [0u8; 4];
        for i in 0..self.cb {
            b[i] = (u >> (8 * (self.cb - 1 - i))) as u8;
        }
        b
    }

    /// Advance past a coded byte. `pos` is the coded length so far, `buf` the
    /// coded bytes.
    pub fn locate(&mut self, pos: usize, buf: &[u8]) {
        if pos > self.start && pos <= self.end && (pos - self.start) % self.cb == 0 {
            // a residual just completed
            let mut u: u64 = 0;
            for &b in &buf[pos - self.cb..pos] {
                u = (u << 8) | b as u64;
            }
            let r = wrap(u as i64 - (1i64 << (self.cbits - 1)), self.bits);
            let x = wrap(self.pred + r, self.bits);
            self.model.update(x, r);
            self.out.push(x as i32);
            self.s += 1;
            if self.s < self.lay.count {
                self.prepare();
            }
        }
        self.active = pos >= self.start && pos < self.end;
        if self.active {
            self.k = (pos - self.start) % self.cb;
            self.known = 0;
            for &b in &buf[pos - self.k..pos] {
                self.known = (self.known << 8) | b as u64;
            }
        }
    }

    /// Byte-level contexts for the CM's slot models: the model's own
    /// contexts, crossed with what is already known of this residual.
    pub fn slot_ctx(&self, out: &mut [u32]) {
        let k = self.k as u32;
        let kn = self.known as u32;
        for (i, &v) in self.prep.sctx.iter().enumerate() {
            out[i] = ((i as u32 + 1).wrapping_mul(0x9e37_79b1) ^ v.wrapping_mul(0x85eb_ca6b) ^ kn.wrapping_mul(0xc2b2_ae35))
                .wrapping_add(k.wrapping_mul(0x27d4_eb2f));
        }
    }

    /// SSE context: loudness, bit position, and where the residual stands
    /// relative to the main prediction. Valid after inputs().
    pub fn sse_ctx(&self) -> usize {
        ((self.prep.errlog << 9) | (self.v0 << 5) | self.pos.min(31)) as usize
    }

    /// Mixer inputs this front-end fills (the rest stay zero).
    pub fn live_inputs(&self) -> usize {
        // value contexts sit after the largest map block, so the count of
        // maps in use doesn't move them
        if self.vtab.is_empty() { NPAR + self.np * MAPS_PER } else { FRONT_IN }
    }

    /// Replacement selectors for mixers 0, 1, 3, 4 (images), with the bit
    /// position folded into the first.
    pub fn mixer_sels(&self, bitpos: u32) -> Option<[usize; 4]> {
        let m = self.prep.msel;
        // the bits of the value already coded matter as much as the context
        let c0 = (self.partial_bits as usize | 1 << bitpos) & 0xff;
        (m[0] != 0).then(|| [((m[0] as usize) >> 3) << 8 | c0, m[1] as usize, m[2] as usize, m[3] as usize])
    }

    /// SSE contexts for images: component with the partial value, the best
    /// predictor's map position, and the main prediction's map position.
    pub fn apm_ctxs(&self, bitpos: u32) -> Option<[usize; 3]> {
        let m = self.prep.msel;
        (m[0] != 0).then(|| {
            let c0 = (self.partial_bits as usize | 1 << bitpos) & 0xff;
            [
                c0,
                ((m[1] as usize) << 8 | c0) & 0xffff,
                ((self.prep.errlog as usize) << 13 | (self.v0 as usize) << 5 | self.pos.min(31) as usize) & 0xffff,
            ]
        })
    }

    /// Mixer weight-set selector: loudness and bit position.
    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        let pos = self.k as u32 * 8 + bitpos;
        ((self.prep.errlog << 5) | pos.min(31)) as usize
    }

    /// Per-bit inputs: for each predictor, its guess relative to the lower end
    /// of the interval the residual is now known to lie in.
    #[inline]
    pub fn inputs(&mut self, c0: u32, bitpos: u32, st: &mut [i32]) {
        let stab = stretch_tab();
        let pos = self.k as u32 * 8 + bitpos;
        self.pos = pos;
        let partial = (self.known << bitpos) | (c0 as u64 - (1u64 << bitpos));
        self.partial_bits = (c0 - (1 << bitpos)) as u32;
        let lo = ((partial << (self.cbits - pos)) as i64) - (1i64 << (self.cbits - 1));
        for i in 0..self.np {
            for j in 0..MAPS_PER {
                let n = MAP_BITS[j];
                let res = if j == 3 { self.d1[i] - lo } else { self.d[i] - lo };
                let sh = (self.cbits + 1).saturating_sub(n).saturating_sub(pos);
                let v = clampn(res >> sh, n);
                let m = i * MAPS_PER + j;
                if m == 2 {
                    self.v0 = v as u32; // main predictor, coarse: where the residual stands
                }
                let ix = if self.vtab.is_empty() {
                    self.map_off[m] + ((v << 5) as usize | pos as usize)
                } else {
                    let key = (m as u32) << 24 ^ (v as u32) << 5 ^ pos ^ (self.prep.pctx[i] as u32) << 18;
                    (key.wrapping_mul(0x9e37_79b1) >> (32 - HMAP_BITS)) as usize
                };
                self.idx[m] = ix;
                st[NPAR + m] = stab[(self.maps[ix] >> 20) as usize];
            }
        }
        // parametric: Laplace around the main prediction at two scales, and
        // around two alternative predictors
        let w = 1i64 << (self.cbits - pos);
        let (lo_f, mid_f, hi_f) = (lo as f64, (lo + w / 2) as f64, (lo + w) as f64);
        let o = &self.prep;
        let par = [
            (self.d[0] as f64, o.scale[0]),
            (self.d[0] as f64, o.scale[1]),
            (self.d[1] as f64, o.scale[0]),
            (self.d1[0] as f64, o.scale[0]),
        ];
        for (j, &(mu, sc)) in par.iter().enumerate() {
            // a Laplace with mean |x| = b; floor keeps silence from going certain
            let b = sc.max(0.3);
            let p1 = laplace_upper(lo_f, mid_f, hi_f, mu, b);
            let p12 = ((p1 * 4096.0) as i32).clamp(1, 4095);
            st[j] = stab[p12 as usize];
        }
        // value-domain contexts: each with the coded bits so far
        if !self.vtab.is_empty() {
            let base = NPAR + NPMAX * MAPS_PER;
            let kb = partial as u32 | 1 << pos;
            for j in 0..o.nv {
                let h = (o.vctx[j] ^ kb.wrapping_mul(0x9e37_79b1)).wrapping_add(j as u32).wrapping_mul(0x2545_f491);
                let ix = (h >> (32 - VTAB_BITS)) as usize;
                self.vidx[j] = ix;
                st[base + 2 * j] = stab[(self.vtab[ix] >> 20) as usize];
                let hs = self.vhist[ix] as usize;
                st[base + 2 * j + 1] = stab[(self.vsm[(j << 8) | hs] >> 20) as usize];
            }
        }
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        for m in 0..self.np * MAPS_PER {
            let lim = MAP_LIMIT[m % MAPS_PER];
            let v = &mut self.maps[self.idx[m]];
            let n = *v & 1023;
            let p22 = (*v >> 10) as i32;
            let rate = RATE_TAB[n as usize];
            let err = (((bit as i32) << 22) - p22) as i64;
            let p22 = (p22 + ((err * rate as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
            *v = (p22 << 10) | if n < lim { n + 1 } else { n };
        }
        if !self.vtab.is_empty() {
            for j in 0..self.prep.nv {
                let ix = self.vidx[j];
                let v = &mut self.vtab[ix];
                let n = *v & 1023;
                let p22 = (*v >> 10) as i32;
                let err = (((bit as i32) << 22) - p22) as i64;
                let p22 = (p22 + ((err * RATE_TAB[n as usize] as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
                *v = (p22 << 10) | if n < 255 { n + 1 } else { n };
                let hs = self.vhist[ix] as usize;
                let c = &mut self.vsm[(j << 8) | hs];
                *c = sm_update(*c, bit);
                self.vhist[ix] = self.stt.next[hs][bit as usize];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The front-ends the predictor can host.
// ---------------------------------------------------------------------------

pub enum Front {
    Samples(SampleFront),
    Jpeg(Box<super::jpeg::JpegFront>),
}

impl Front {
    pub fn new(lay: Layout, start: usize, buf: &[u8]) -> Self {
        if lay.kind == KIND_JPEG {
            Front::Jpeg(Box::new(super::jpeg::JpegFront::new(buf, start, lay.count, lay.stride)))
        } else {
            Front::Samples(SampleFront::new(lay, start, buf))
        }
    }
    #[inline]
    pub fn active(&self) -> bool {
        match self {
            Front::Samples(f) => f.active,
            Front::Jpeg(f) => f.active,
        }
    }
    pub fn finished(&self, pos: usize) -> bool {
        match self {
            Front::Samples(f) => f.finished(),
            Front::Jpeg(f) => pos >= f.region_end(),
        }
    }
    pub fn locate(&mut self, pos: usize, buf: &[u8]) {
        match self {
            Front::Samples(f) => f.locate(pos, buf),
            Front::Jpeg(f) => f.locate(pos, buf),
        }
    }
    pub fn live_inputs(&self) -> usize {
        match self {
            Front::Samples(f) => f.live_inputs(),
            Front::Jpeg(_) => super::jpeg::JPEG_IN,
        }
    }
    #[inline]
    pub fn inputs(&mut self, c0: u32, bitpos: u32, st: &mut [i32]) {
        match self {
            Front::Samples(f) => f.inputs(c0, bitpos, st),
            Front::Jpeg(f) => f.inputs(c0, bitpos, st),
        }
    }
    #[inline]
    pub fn update(&mut self, bit: u32) {
        match self {
            Front::Samples(f) => f.update(bit),
            Front::Jpeg(f) => f.update(bit),
        }
    }
    /// Byte-level slot contexts, for front-ends that supply them.
    pub fn slot_ctx(&self, out: &mut [u32]) -> bool {
        match self {
            Front::Samples(f) => {
                f.slot_ctx(out);
                true
            }
            Front::Jpeg(_) => false,
        }
    }
    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        match self {
            Front::Samples(f) => f.mixer_sel(bitpos),
            Front::Jpeg(f) => f.mixer_sel(bitpos),
        }
    }
    pub fn sse_ctx(&self) -> usize {
        match self {
            Front::Samples(f) => f.sse_ctx(),
            Front::Jpeg(f) => f.sse_ctx(),
        }
    }
    /// Mixer selectors from a sample model (images), if it supplies them.
    pub fn sample_sels(&self, bitpos: u32) -> Option<[usize; 4]> {
        match self {
            Front::Samples(f) => f.mixer_sels(bitpos),
            Front::Jpeg(_) => None,
        }
    }

    /// Replacement contexts for the first three SSE stages, if this front-end has them.
    pub fn apm_ctxs(&self) -> Option<[usize; 3]> {
        match self {
            Front::Samples(f) => f.apm_ctxs(f.pos & 7),
            Front::Jpeg(f) => f.data_active().then(|| f.apm_ctxs()),
        }
    }

    /// Replacement selectors for mixers 0, 1, 3 and 4, if this front-end has them.
    pub fn mixer_sels(&self) -> Option<[usize; 4]> {
        match self {
            Front::Samples(_) => None,
            Front::Jpeg(f) => f.data_active().then(|| f.mixer_sels()),
        }
    }
    /// Samples decoded in this region (empty for passthrough front-ends).
    pub fn into_samples(self) -> Vec<i32> {
        match self {
            Front::Samples(f) => f.out,
            Front::Jpeg(_) => Vec::new(),
        }
    }
    /// Encoder side: the coded bytes for sample `x`.
    pub fn code(&self, x: i64) -> [u8; 4] {
        match self {
            Front::Samples(f) => f.code(x),
            Front::Jpeg(_) => unreachable!("JPEG regions pass through"),
        }
    }
}

