//! JPEG model: predicts the entropy-coded scan data of baseline JPEGs.
//!
//! A JPEG's scan is Huffman-coded DCT coefficients — already compressed, and
//! noise to a byte-level model. But the Huffman coding is a weak model of its
//! own: it codes each coefficient knowing nothing of the blocks around it.
//! This model decodes the scan bit by bit as it is coded, so at every bit it
//! knows exactly what is being decoded — which Huffman code, which
//! coefficient of which block — and it conditions that bit on the same
//! coefficient in the neighbouring blocks, which a real image makes strongly
//! correlated. The bytes pass through unchanged, so there is nothing to
//! reconstruct: the decoder sees the same bits and decodes them the same way.
//!
//! Only baseline and extended (8-bit, Huffman) sequential scans are modelled.
//! Anything the parser does not understand turns the model off for the rest of
//! the scan; the bytes still flow through the ordinary models.

use super::{sm_init, sm_update, state_tab, stretch_tab, StateTab, RATE_TAB};

/// Hashed contexts per bit.
const NCTX: usize = 24;
/// Mixer inputs: a direct counter and a bit history per context, plus the
/// special-bit oracle.
pub const JPEG_IN: usize = 2 * NCTX + 1;
const TABLE_BITS: u32 = 22;

const ZZU: [u8; 64] = [
    0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4,
    5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7,
];
const ZZV: [u8; 64] = [
    0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4,
    3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 7,
];

/// (u + 8v) -> zigzag index.
const ZPOS: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[ZZU[i] as usize + 8 * ZZV[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// 16 * log2(n), roughly, in integers: what paq8px calls ilog.
#[inline]
fn ilog16(n: u32) -> i32 {
    if n == 0 {
        return 0;
    }
    let b = 31 - n.leading_zeros();
    // the four bits after the leading one interpolate within the octave
    let frac = if b >= 4 { (n >> (b - 4)) & 15 } else { (n << (4 - b)) & 15 };
    (b * 16 + frac) as i32
}

#[inline]
fn slog(v: i64) -> i32 {
    let m = ilog16(v.unsigned_abs().min(u32::MAX as u64) as u32 + 1);
    if v < 0 { -m } else { m }
}

/// Weight of a frequency's basis function at the block edge, x16 (paq8px).
#[inline]
fn edge_w(f: u8) -> i64 {
    if f != 0 { 16 * (16 + f as i64) } else { 185 }
}

/// A canonical Huffman table for bit-serial decoding.
#[derive(Clone)]
struct Huff {
    maxcode: [i32; 17], // largest code of each length, -1 if none
    valptr: [i32; 17],  // index of that length's first symbol
    mincode: [i32; 17],
    vals: Vec<u8>,
}

impl Huff {
    fn new(counts: &[u8; 16], vals: &[u8]) -> Option<Huff> {
        let mut h = Huff { maxcode: [-1; 17], valptr: [0; 17], mincode: [0; 17], vals: vals.to_vec() };
        let (mut code, mut k) = (0i32, 0i32);
        for l in 1..=16 {
            let n = counts[l - 1] as i32;
            if n > 0 {
                h.valptr[l] = k;
                h.mincode[l] = code;
                code += n;
                k += n;
                h.maxcode[l] = code - 1;
            }
            if code > (1 << l) {
                return None;
            }
            code <<= 1;
        }
        (k as usize <= vals.len()).then_some(h)
    }

    /// The symbol for (code, len), if it is a complete code.
    #[inline]
    fn lookup(&self, code: i32, len: usize) -> Option<u8> {
        if len <= 16 && self.maxcode[len] >= 0 && code <= self.maxcode[len] && code >= self.mincode[len] {
            self.vals.get((self.valptr[len] + code - self.mincode[len]) as usize).copied()
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    bw: usize, // blocks across, in this component's grid
    bh: usize,
}

/// What the scan needs from the headers.
struct Scan {
    comps: Vec<Comp>,       // all frame components
    scomp: Vec<usize>,      // frame component index of each scan component
    td: Vec<usize>,         // DC table per scan component
    ta: Vec<usize>,         // AC table per scan component
    dc: Vec<Option<Huff>>,  // 4 DC tables
    ac: Vec<Option<Huff>>,  // 4 AC tables
    qt: [[u16; 64]; 4],
    restart: usize,         // MCUs per restart interval, 0 = none
    mcux: usize,
    mcuy: usize,
    // block order inside one MCU: (scan component, dx, dy)
    mcu_blocks: Vec<(usize, usize, usize)>,
}

fn u16be(d: &[u8], p: usize) -> Option<usize> {
    Some(((*d.get(p)? as usize) << 8) | *d.get(p + 1)? as usize)
}

/// Parse markers from SOI up to the SOS whose header ends at `scan_at`.
fn parse_headers(d: &[u8], scan_at: usize) -> Option<Scan> {
    if d.get(0..2)? != [0xff, 0xd8] {
        return None;
    }
    let mut p = 2;
    let mut dc: Vec<Option<Huff>> = vec![None, None, None, None];
    let mut ac: Vec<Option<Huff>> = vec![None, None, None, None];
    let mut qt = [[1u16; 64]; 4];
    let mut comps: Vec<Comp> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut restart = 0;
    let mut last_sos: Option<(Vec<usize>, Vec<usize>, Vec<usize>)> = None;
    while p + 4 <= scan_at {
        if d[p] != 0xff {
            // inside an earlier scan's data: skip to the next real marker
            p += 1;
            continue;
        }
        let m = d[p + 1];
        if m == 0xff {
            p += 1;
            continue;
        }
        if m == 0x00 || (0xd0..=0xd7).contains(&m) || m == 0x01 {
            p += 2;
            continue;
        }
        let len = u16be(d, p + 2)?;
        let seg = d.get(p + 4..p + 2 + len)?;
        match m {
            0xc0 | 0xc1 => {
                if seg[0] != 8 {
                    return None;
                }
                height = u16be(seg, 1)?;
                width = u16be(seg, 3)?;
                let nf = seg[5] as usize;
                comps.clear();
                for i in 0..nf {
                    let b = seg.get(6 + 3 * i..9 + 3 * i)?;
                    let (h, v) = ((b[1] >> 4) as usize, (b[1] & 15) as usize);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || b[2] > 3 {
                        return None;
                    }
                    comps.push(Comp { id: b[0], h, v, tq: b[2] as usize, bw: 0, bh: 0 });
                }
            }
            0xc2..=0xc3 | 0xc5..=0xcb | 0xcd..=0xcf => return None, // progressive, lossless, arithmetic
            0xc4 => {
                let mut q = 0;
                while q + 17 <= seg.len() {
                    let (tc, th) = (seg[q] >> 4, (seg[q] & 15) as usize);
                    let counts: [u8; 16] = seg[q + 1..q + 17].try_into().ok()?;
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    let vals = seg.get(q + 17..q + 17 + n)?;
                    if th > 3 || tc > 1 {
                        return None;
                    }
                    let t = Huff::new(&counts, vals)?;
                    if tc == 0 { dc[th] = Some(t) } else { ac[th] = Some(t) }
                    q += 17 + n;
                }
            }
            0xdb => {
                let mut q = 0;
                while q < seg.len() {
                    let (pq, tq) = (seg[q] >> 4, (seg[q] & 15) as usize);
                    if tq > 3 {
                        return None;
                    }
                    for k in 0..64 {
                        qt[tq][k] = if pq == 0 { *seg.get(q + 1 + k)? as u16 } else { u16be(seg, q + 1 + 2 * k)? as u16 };
                    }
                    q += 1 + 64 * (1 + pq as usize);
                }
            }
            0xdd => restart = u16be(seg, 0)?,
            0xda => {
                let ns = seg[0] as usize;
                let (mut sc, mut td, mut ta) = (Vec::new(), Vec::new(), Vec::new());
                for i in 0..ns {
                    let b = seg.get(1 + 2 * i..3 + 2 * i)?;
                    sc.push(comps.iter().position(|c| c.id == b[0])?);
                    td.push((b[1] >> 4) as usize);
                    ta.push((b[1] & 15) as usize);
                }
                let tail = seg.get(1 + 2 * ns..4 + 2 * ns)?;
                if tail != [0, 63, 0] {
                    return None; // not a sequential scan
                }
                last_sos = Some((sc, td, ta));
                if p + 2 + len == scan_at {
                    break;
                }
            }
            0xd9 => return None,
            _ => {}
        }
        p += 2 + len;
    }
    let (scomp, td, ta) = last_sos?;
    if comps.is_empty() || width == 0 || height == 0 || td.iter().chain(&ta).any(|&t| t > 3) {
        return None;
    }
    let hmax = comps.iter().map(|c| c.h).max()?;
    let vmax = comps.iter().map(|c| c.v).max()?;
    let mcux = width.div_ceil(8 * hmax);
    let mcuy = height.div_ceil(8 * vmax);
    for c in comps.iter_mut() {
        c.bw = mcux * c.h;
        c.bh = mcuy * c.v;
    }
    let mut mcu_blocks = Vec::new();
    let (mcux, mcuy) = if scomp.len() == 1 {
        // a non-interleaved scan walks the component's own grid, block by block
        let c = &comps[scomp[0]];
        let bw = (width * c.h).div_ceil(hmax).div_ceil(8);
        let bh = (height * c.v).div_ceil(vmax).div_ceil(8);
        mcu_blocks.push((0, 0, 0));
        (bw, bh)
    } else {
        for (i, &fc) in scomp.iter().enumerate() {
            for dy in 0..comps[fc].v {
                for dx in 0..comps[fc].h {
                    mcu_blocks.push((i, dx, dy));
                }
            }
        }
        if mcu_blocks.len() > 10 {
            return None;
        }
        (mcux, mcuy)
    };
    if mcux * mcuy == 0 || mcux * mcuy > 1 << 24 {
        return None;
    }
    Some(Scan { comps, scomp, td, ta, dc, ac, qt, restart, mcux, mcuy, mcu_blocks })
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Code,         // reading a Huffman code
    Extra(usize), // reading this many extra bits of a coefficient
    Pad,          // padding bits before a restart marker or the end
    Done,         // not modelling (finished, or gave up)
}

/// Signed log bucket of a coefficient.
#[inline]
fn lg(v: i32) -> u32 {
    let m = 32 - v.unsigned_abs().leading_zeros();
    (m << 1) | (v < 0) as u32
}

pub struct JpegFront {
    scan: Option<Scan>,
    start: usize,
    end: usize,
    pub active: bool,
    // coefficient memory: per frame component, all blocks, 64 each
    coef: Vec<Vec<i16>>,
    // decoding position
    mcu: usize,
    blk: usize, // block within the MCU
    k: usize,   // zigzag index of the next coefficient
    cur: [i32; 64],
    dc_pred: Vec<i32>,
    phase: Phase,
    code: i32,
    clen: usize,
    extra: i32,
    elen: usize,
    is_dc: bool,
    run_coef: usize, // coefficient the pending extra bits belong to
    nz: u32,         // non-zero coefficients so far in this block
    zrun: u32,       // zeros since the last non-zero
    // byte bookkeeping
    stuff: bool,           // the byte being coded is a stuffing 0x00
    marker: u8,            // 2: at the FF of a restart marker, 1: at its second byte
    rst_n: u8,
    mcus_left: usize,      // until the next restart
    bitpos: u32,
    byte_so_far: u32,
    // model
    table: Vec<u32>,
    hist: Vec<u8>, // bit-history state per table cell
    sm: Vec<u32>,  // per context: state -> P(1)
    stt: &'static StateTab,
    idx: [usize; NCTX],
    special: Vec<u32>,
    sidx: usize,
    sexp: Option<u32>, // the bit a special position must have
    ctxbase: [u32; NCTX],
    pub sel: usize,
    // cross-block prediction: the north and west blocks projected onto the
    // shared edges, minus what this block has decoded so far
    sum_u: [i64; 8],
    sum_v: [i64; 8],
    adv: [i32; 3],     // predictions from west, both, north (log scale)
    pred_cat: u32,     // the size category adv[1] implies
    prev_blk: [i32; 64], // the block decoded before this one, and the one before
    prev_blk2: [i32; 64],
    sum_abs: i32,      // sum of |coefficient| so far in this block
}

impl JpegFront {
    /// `buf` holds every byte before the scan; the JPEG starts `back` bytes
    /// before `start`.
    pub fn new(buf: &[u8], start: usize, len: usize, back: usize) -> Self {
        let scan = start.checked_sub(back).and_then(|s0| parse_headers(&buf[s0..], back));
        let coef = match &scan {
            Some(s) => s.comps.iter().map(|c| vec![0i16; c.bw * c.bh * 64]).collect(),
            None => Vec::new(),
        };
        let ncomp = scan.as_ref().map_or(0, |s| s.scomp.len());
        let restart = scan.as_ref().map_or(0, |s| s.restart);
        let mut f = Self {
            phase: if scan.is_some() { Phase::Code } else { Phase::Done },
            scan,
            start,
            end: start + len,
            active: false,
            coef,
            mcu: 0,
            blk: 0,
            k: 0,
            cur: [0; 64],
            dc_pred: vec![0; ncomp],
            code: 0,
            clen: 0,
            extra: 0,
            elen: 0,
            is_dc: true,
            run_coef: 0,
            nz: 0,
            zrun: 0,
            stuff: false,
            marker: 0,
            rst_n: 0,
            mcus_left: restart,
            bitpos: 0,
            byte_so_far: 1,
            table: vec![(1 << 21) << 10; 1 << TABLE_BITS],
            hist: vec![0; 1 << TABLE_BITS],
            sm: (0..NCTX).flat_map(|_| sm_init(state_tab())).collect(),
            stt: state_tab(),
            idx: [0; NCTX],
            special: vec![(1 << 21) << 10; 4096],
            sidx: 0,
            sexp: None,
            ctxbase: [0; NCTX],
            sel: 0,
            sum_u: [0; 8],
            sum_v: [0; 8],
            adv: [0; 3],
            pred_cat: 0,
            prev_blk: [0; 64],
            prev_blk2: [0; 64],
            sum_abs: 0,
        };
        f.start_block();
        f.active = f.start < f.end;
        f
    }

    pub fn region_end(&self) -> usize {
        self.end
    }

    /// Position of block `(component, x, y)` in the coefficient memory.
    #[inline]
    fn block_at(&self, fc: usize, bx: usize, by: usize) -> usize {
        let s = self.scan.as_ref().unwrap();
        (by * s.comps[fc].bw + bx) * 64
    }

    /// (frame component, block x, block y) of the current block.
    fn cur_block(&self) -> (usize, usize, usize) {
        let s = self.scan.as_ref().unwrap();
        let (sc, dx, dy) = s.mcu_blocks[self.blk];
        let fc = s.scomp[sc];
        if s.scomp.len() == 1 {
            (fc, self.mcu % s.mcux, self.mcu / s.mcux)
        } else {
            let (mx, my) = (self.mcu % s.mcux, self.mcu / s.mcux);
            (fc, mx * s.comps[fc].h + dx, my * s.comps[fc].v + dy)
        }
    }

    /// Called at each byte boundary with the coded stream so far.
    pub fn locate(&mut self, pos: usize, buf: &[u8]) {
        if pos <= self.start {
            return;
        }
        self.active = pos < self.end;
        let b = buf[pos - 1];
        // what the byte just finished was decides what the next one is
        if self.marker == 2 {
            self.marker = 1;
        } else if self.marker == 1 {
            self.marker = 0;
            if b != 0xd0 + self.rst_n {
                self.phase = Phase::Done;
            }
            self.rst_n = (self.rst_n + 1) & 7;
            self.restart_reset();
        } else if self.stuff {
            self.stuff = false;
            if b != 0 {
                self.phase = Phase::Done;
            }
        } else if b == 0xff {
            self.stuff = true;
        }
        self.bitpos = 0;
        self.byte_so_far = 1;
        // after padding completes a byte, a due restart marker follows
        if self.phase == Phase::Pad && !self.stuff && self.marker == 0 {
            let s = self.scan.as_ref().unwrap();
            if s.restart > 0 && self.mcu < s.mcux * s.mcuy {
                self.marker = 2;
            } else {
                self.phase = Phase::Done;
            }
        }
    }

    fn restart_reset(&mut self) {
        if let Some(s) = &self.scan {
            self.mcus_left = s.restart;
            self.dc_pred.iter_mut().for_each(|d| *d = 0);
            if self.phase != Phase::Done {
                self.phase = Phase::Code;
                self.code = 0;
                self.clen = 0;
                self.block_ctx(); // the DC prediction just changed
            }
        }
    }

    /// Is the decoder following the scan (rather than finished or lost)?
    pub fn data_active(&self) -> bool {
        self.active && matches!(self.phase, Phase::Code | Phase::Extra(_))
    }

    /// Is this bit an entropy-coded data bit?
    #[inline]
    fn data_bit(&self) -> bool {
        !self.stuff && self.marker == 0 && matches!(self.phase, Phase::Code | Phase::Extra(_))
    }

    /// At a block's start: project the north and west blocks (dequantized)
    /// onto the edges this block shares with them.
    fn start_block(&mut self) {
        let Some(s) = &self.scan else { return };
        let (fc, bx, by) = self.cur_block();
        let q = &s.qt[s.comps[fc].tq];
        let bw = s.comps[fc].bw;
        let c = &self.coef[fc];
        self.sum_u = [0; 8];
        self.sum_v = [0; 8];
        if by > 0 {
            let n = ((by - 1) * bw + bx) * 64;
            for i in 0..64 {
                let sign = if ZZV[i] & 1 != 0 { -1 } else { 1 };
                self.sum_u[ZZU[i] as usize] += sign * edge_w(ZZV[i]) * q[i] as i64 * c[n + i] as i64;
            }
        }
        if bx > 0 {
            let w = (by * bw + bx - 1) * 64;
            for i in 0..64 {
                let sign = if ZZU[i] & 1 != 0 { -1 } else { 1 };
                self.sum_v[ZZV[i] as usize] += sign * edge_w(ZZU[i]) * q[i] as i64 * c[w + i] as i64;
            }
        }
        self.sum_abs = 0;
        self.block_ctx();
    }

    /// A coefficient of this block became known: take it out of the edge sums.
    fn known_coef(&mut self, j: usize, v: i32) {
        let Some(s) = &self.scan else { return };
        let (fc, _, _) = self.cur_block();
        let q = s.qt[s.comps[fc].tq][j] as i64;
        self.sum_u[ZZU[j] as usize] -= edge_w(ZZV[j]) * q * v as i64;
        self.sum_v[ZZV[j] as usize] -= edge_w(ZZU[j]) * q * v as i64;
        self.sum_abs += v.abs();
    }

    /// Contexts that hold for the whole of the current coefficient position.
    fn block_ctx(&mut self) {
        let Some(s) = &self.scan else { return };
        let (fc, bx, by) = self.cur_block();
        let k = self.k.min(63);
        let qt = &s.qt[s.comps[fc].tq];
        let at = |bx: usize, by: usize| -> Option<usize> {
            (bx < s.comps[fc].bw && by < s.comps[fc].bh).then(|| (by * s.comps[fc].bw + bx) * 64)
        };
        let c = &self.coef[fc];
        let above = if by > 0 { at(bx, by - 1) } else { None };
        let left = if bx > 0 { at(bx - 1, by) } else { None };
        let get = |b: Option<usize>, k: usize| b.map_or(0, |b| c[b + k] as i32);
        let (a, l) = (get(above, k), get(left, k));
        let energy = |b: Option<usize>| b.map_or(0u32, |b| (k..64).map(|j| c[b + j].unsigned_abs() as u32).sum());
        let nzc = |b: Option<usize>| b.map_or(0u32, |b| (0..64).filter(|&j| c[b + j] != 0).count() as u32);
        let el = 32 - (energy(above) + energy(left)).leading_zeros();
        let (u, v) = (ZZU[k], ZZV[k]);
        // edge predictions: from the west edge, both, and the north edge
        let den = (qt[k] as i64 * 185 * (16 + v as i64) * (16 + u as i64) / 128).max(1);
        let dc_prev = if k == 0 { self.dc_pred.get(self.blk_sc()).copied().unwrap_or(0) as i64 } else { 0 };
        for i in 0..3 {
            let p = (self.sum_u[u as usize] * i as i64 + self.sum_v[v as usize] * (2 - i as i64)) / den;
            self.adv[i] = slog(p - dc_prev);
        }
        if left.is_none() {
            self.adv[1] = self.adv[2];
            self.adv[0] = 0;
        }
        if above.is_none() {
            self.adv[1] = self.adv[0];
            self.adv[2] = 0;
        }
        self.pred_cat = ((self.adv[1].unsigned_abs() + 15) >> 4).min(15);
        // in-block neighbours, rescaled to this coefficient's quantiser
        let lcp = |du: u8, dv: u8| -> i32 {
            if u < du || v < dv {
                return 0;
            }
            let j = ZPOS[(u - du) as usize + 8 * (v - dv) as usize] as usize;
            if j >= k {
                return 0;
            }
            slog(self.cur[j] as i64 * qt[j] as i64 / qt[k].max(1) as i64)
        };
        let (l0, l1, l2, l3, l4) = (lcp(1, 0), lcp(0, 1), lcp(2, 0), lcp(0, 2), lcp(1, 1));
        let prev = self.prev_blk[k];
        let prev2 = self.prev_blk2[k];
        let kk = k as u32 | (self.is_dc as u32) << 6 | (fc.min(3) as u32) << 7;
        let fcu = fc as u32;
        let h = |tag: u32, a: i32, b: i32, c: i32| {
            tag.wrapping_mul(0x9e37_79b1)
                ^ (a as u32).wrapping_mul(0x85eb_ca6b)
                ^ (b as u32).wrapping_mul(0xc2b2_ae35)
                ^ (c as u32).wrapping_mul(0x1656_67b1)
                ^ kk.wrapping_mul(0x27d4_eb2f)
        };
        let clamp = |v: i32| v.clamp(-31, 31);
        let (zu, zv) = (u as i32, v as i32);
        let low = (zu + zv < 4) as i32;
        let ad = self.adv;
        self.ctxbase = [
            h(1, fcu as i32, 0, 0),
            h(2, self.sum_abs.min(255) >> 2, lg(prev) as i32, 0),
            h(3, ad[1] / 17, l0.max(l1) / 24, l2 / 20 + (l3 / 24 << 8)),
            h(4, ad[1] / 11, l0 / 50, l1 / 50),
            h(5, ad[2] / 13, prev / 11, low),
            h(6, ad[0] / 13, prev / 11, low),
            h(7, ad[2] / 13, prev / 40, prev2 / 40),
            h(8, ad[0] / 13, prev / 40, prev2 / 40),
            h(9, l0 / 12, l1 / 12, l4 / 10),
            h(10, zu / 2, prev / 40, prev2 / 28),
            h(11, zv / 2, prev / 40, prev2 / 28),
            h(12, (bx >> 3) as i32, (zu + zv).min(7), prev / 40),
            h(13, clamp(a), 0, 0),
            h(14, clamp(l), 0, 0),
            h(15, self.nz as i32, self.zrun.min(31) as i32, 0),
            h(16, ad[1] / 8, 0, 0),
            h(17, self.pred_cat as i32, 0, 0),
            h(18, lg(a) as i32, lg(l) as i32, 0),
            h(19, el as i32, fcu as i32, 0),
            h(20, ad[1] / 16, ad[2] / 16 - ad[0] / 16, 0),
            h(21, nzc(above) as i32, nzc(left) as i32, 0),
            h(22, lg(if k > 0 { self.cur[k - 1] } else { 0 }) as i32, 0, 0),
            h(23, ad[1] / 24, self.sum_abs.min(255) >> 4, 0),
            h(24, l0 / 16, l1 / 16, ad[1] / 24),
        ];
        self.sel = ((k.min(15) as usize) << 1) | self.is_dc as usize;
    }

    /// For the current bit, which way the edge prediction leans: two bits.
    fn pred_bits(&self) -> u32 {
        match self.phase {
            Phase::Code => self.pred_cat.min(3),
            Phase::Extra(n) => {
                let pm = self.adv[1].unsigned_abs() as i32;
                if self.elen == 0 {
                    ((self.adv[1] > 0) as u32) << 1 | (self.pred_cat >= n as u32) as u32
                } else {
                    let positive = (self.extra >> (self.elen - 1)) & 1 == 1;
                    let x = self.extra as u32 & ((1 << self.elen) - 1);
                    let mid = (2 * x + 1) << (n - self.elen - 1);
                    let d = if positive { pm - ilog16(mid + 1) } else { ilog16((1 << n) - mid) - pm };
                    ((d >= 0) as u32) << 1 | (d.abs() >= 8) as u32
                }
            }
            _ => 0,
        }
    }

    fn blk_sc(&self) -> usize {
        self.scan.as_ref().map_or(0, |s| s.mcu_blocks[self.blk].0)
    }

    /// Per-bit inputs into `st` (JPEG_IN of them).
    #[inline]
    pub fn inputs(&mut self, c0: u32, bitpos: u32, st: &mut [i32]) {
        let stab = stretch_tab();
        self.sexp = None;
        if !self.data_bit() {
            // stuffing, marker and padding bits are nearly certain
            let exp = if self.stuff {
                Some(0)
            } else if self.marker == 2 {
                Some(1)
            } else if self.marker == 1 {
                Some(((0xd0 + self.rst_n as u32) >> (7 - bitpos)) & 1)
            } else if self.phase == Phase::Pad {
                Some(1)
            } else {
                None
            };
            st[..2 * NCTX].fill(0);
            match exp {
                Some(e) => {
                    self.sexp = Some(e);
                    self.sidx = ((bitpos as usize) << 3 | (self.stuff as usize) << 2 | (self.marker as usize)) << 1 | e as usize;
                    st[2 * NCTX] = stab[(self.special[self.sidx] >> 20) as usize];
                }
                None => st[2 * NCTX] = 0,
            }
            let _ = c0;
            return;
        }
        st[2 * NCTX] = 0;
        // the state inside the current symbol
        let ph = match self.phase {
            Phase::Code => (self.clen as u32) << 16 | self.code as u32,
            Phase::Extra(n) => 1 << 24 | (n as u32) << 20 | (self.elen as u32) << 16 | self.extra as u32,
            _ => 0,
        };
        let mask = (1usize << TABLE_BITS) - 1;
        let pb = self.pred_bits();
        for i in 0..NCTX {
            // the even contexts also see which way the edge prediction leans
            let ph = if i & 1 == 0 { ph ^ pb << 28 } else { ph };
            let hsh = self.ctxbase[i] ^ ph.wrapping_mul(0x6c8e_9cf5).wrapping_add(i as u32 * 0x9e37_79b9);
            let ix = (hsh.wrapping_mul(0x2545_f491) >> (32 - TABLE_BITS)) as usize & mask;
            self.idx[i] = ix;
            st[i] = stab[(self.table[ix] >> 20) as usize];
            let h = self.hist[ix] as usize;
            st[NCTX + i] = stab[(self.sm[(i << 8) | h] >> 20) as usize];
        }
    }

    #[inline]
    fn train(v: &mut u32, bit: u32, limit: u32) {
        let n = *v & 1023;
        let p22 = (*v >> 10) as i32;
        let rate = RATE_TAB[n as usize];
        let err = (((bit as i32) << 22) - p22) as i64;
        let p22 = (p22 + ((err * rate as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
        *v = (p22 << 10) | if n < limit { n + 1 } else { n };
    }

    /// Learn from `bit`, then advance the decoder by it.
    pub fn update(&mut self, bit: u32) {
        if let Some(e) = self.sexp.take() {
            Self::train(&mut self.special[self.sidx], bit, 255);
            if bit != e && (self.marker != 0 || self.stuff) {
                self.phase = Phase::Done; // not the marker we expected
            }
        } else if self.data_bit() {
            for i in 0..NCTX {
                let ix = self.idx[i];
                Self::train(&mut self.table[ix], bit, 255);
                let h = self.hist[ix] as usize;
                let c = &mut self.sm[(i << 8) | h];
                *c = sm_update(*c, bit);
                self.hist[ix] = self.stt.next[h][bit as usize];
            }
            self.step(bit);
        }
        self.bitpos += 1;
        self.byte_so_far = (self.byte_so_far << 1) | bit;
    }

    fn give_up(&mut self) {
        self.phase = Phase::Done;
    }

    /// Advance the Huffman decoder by one data bit.
    fn step(&mut self, bit: u32) {
        let Some(s) = &self.scan else { return self.give_up() };
        match self.phase {
            Phase::Code => {
                self.code = (self.code << 1) | bit as i32;
                self.clen += 1;
                let sc = s.mcu_blocks[self.blk].0;
                let t = if self.is_dc { &s.dc[s.td[sc]] } else { &s.ac[s.ta[sc]] };
                let Some(t) = t else { return self.give_up() };
                if let Some(sym) = t.lookup(self.code, self.clen) {
                    self.code = 0;
                    self.clen = 0;
                    if self.is_dc {
                        if sym > 11 {
                            return self.give_up();
                        }
                        self.run_coef = 0;
                        if sym == 0 {
                            self.finish_coef(0);
                        } else {
                            self.phase = Phase::Extra(sym as usize);
                        }
                    } else {
                        let (r, sz) = ((sym >> 4) as usize, (sym & 15) as usize);
                        if sz == 0 {
                            if r == 15 {
                                // sixteen zeros
                                self.k += 16;
                                self.zrun += 16;
                                if self.k > 64 {
                                    return self.give_up();
                                }
                                if self.k == 64 {
                                    self.end_block();
                                } else {
                                    self.block_ctx();
                                }
                            } else if r == 0 {
                                self.end_block(); // EOB
                            } else {
                                return self.give_up();
                            }
                        } else {
                            self.k += r;
                            self.zrun += r as u32;
                            if self.k > 63 {
                                return self.give_up();
                            }
                            self.run_coef = self.k;
                            self.phase = Phase::Extra(sz);
                            self.block_ctx();
                        }
                    }
                } else if self.clen >= 16 {
                    self.give_up();
                }
            }
            Phase::Extra(n) => {
                self.extra = (self.extra << 1) | bit as i32;
                self.elen += 1;
                if self.elen == n {
                    let v = if self.extra < 1 << (n - 1) { self.extra - (1 << n) + 1 } else { self.extra };
                    self.extra = 0;
                    self.elen = 0;
                    self.finish_coef(v);
                }
            }
            _ => {}
        }
    }

    fn finish_coef(&mut self, v: i32) {
        if self.is_dc {
            let sc = self.blk_sc();
            self.dc_pred[sc] += v;
            self.cur[0] = self.dc_pred[sc];
            self.known_coef(0, self.cur[0]);
            self.is_dc = false;
            self.k = 1;
        } else {
            self.cur[self.run_coef] = v;
            if v != 0 {
                self.known_coef(self.run_coef, v);
            }
            self.k = self.run_coef + 1;
        }
        if v != 0 {
            self.nz += 1;
            self.zrun = 0;
        }
        self.phase = Phase::Code;
        if self.k >= 64 {
            self.end_block();
        } else {
            self.block_ctx();
        }
    }

    fn end_block(&mut self) {
        let (fc, bx, by) = self.cur_block();
        let at = self.block_at(fc, bx, by);
        if let Some(dst) = self.coef[fc].get_mut(at..at + 64) {
            for (d, &c) in dst.iter_mut().zip(self.cur.iter()) {
                *d = c.clamp(-32768, 32767) as i16;
            }
        }
        self.prev_blk2 = self.prev_blk;
        self.prev_blk = self.cur;
        self.cur = [0; 64];
        self.k = 0;
        self.nz = 0;
        self.zrun = 0;
        self.is_dc = true;
        self.phase = Phase::Code;
        let s = self.scan.as_ref().unwrap();
        self.blk += 1;
        if self.blk == s.mcu_blocks.len() {
            self.blk = 0;
            self.mcu += 1;
            let total = s.mcux * s.mcuy;
            if self.mcu >= total {
                self.phase = Phase::Pad;
                return;
            }
            if s.restart > 0 {
                self.mcus_left -= 1;
                if self.mcus_left == 0 {
                    self.phase = Phase::Pad;
                    // a restart begins on a byte boundary: if this one already
                    // ends a byte, the marker comes straight away
                    self.start_block();
                    return;
                }
            }
        }
        self.start_block();
    }

    /// Weight-set selectors for the other four mixers inside a scan: the
    /// symbol being decoded, the coefficient, the edge prediction, and the
    /// block's sparsity — byte-oriented selectors mean nothing here.
    pub fn mixer_sels(&self) -> [usize; 4] {
        let sym = match self.phase {
            Phase::Code => ((1usize << self.clen.min(7)) | (self.code as usize & ((1 << self.clen.min(7)) - 1))) & 0x7f,
            Phase::Extra(n) => 0x80 | n.min(15) << 3 | self.elen.min(7),
            _ => 0,
        };
        let comp = self.scan.as_ref().map_or(0, |s| s.mcu_blocks[self.blk].0.min(3));
        [
            sym,
            self.k.min(63) | comp << 6,
            (self.pred_bits() as usize) | (self.pred_cat as usize) << 2 | (self.is_dc as usize) << 6 | (matches!(self.phase, Phase::Extra(_)) as usize) << 7,
            (self.nz.min(31) as usize) << 4 | self.zrun.min(15) as usize,
        ]
    }

    /// SSE contexts for the calibration stages inside a scan (three of them;
    /// the fourth uses sse_ctx).
    pub fn apm_ctxs(&self) -> [usize; 3] {
        let ph = match self.phase {
            Phase::Code => (1usize << self.clen.min(10)) | (self.code as usize & ((1 << self.clen.min(10)) - 1)),
            Phase::Extra(n) => 0x800 | n.min(15) << 4 | self.elen.min(15),
            _ => 0,
        };
        let comp = self.scan.as_ref().map_or(0, |s| s.mcu_blocks[self.blk].0.min(3));
        [
            ph & 0xfff | (self.is_dc as usize) << 12 | (comp.min(1)) << 13,
            (self.k.min(63) << 4 | (self.pred_bits() as usize) << 2 | comp.min(3)) & 0xffff,
            ((self.adv[1].unsigned_abs() as usize).min(255) << 6 | (self.pred_bits() as usize) << 4 | self.nz.min(15) as usize) & 0xffff,
        ]
    }

    /// Padding ends at the byte boundary; nothing to do per bit here.
    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        let ph = match self.phase {
            Phase::Code => 0,
            Phase::Extra(_) => 1,
            _ => 2,
        };
        (self.sel << 5 | ph << 3 | bitpos as usize) & 1023
    }

    pub fn sse_ctx(&self) -> usize {
        let ph = match self.phase {
            Phase::Code => self.clen.min(15),
            Phase::Extra(n) => 16 + n.min(15),
            _ => 32,
        };
        (self.sel << 6 | ph) & 0x3fff
    }
}

/// Analysis and test aid: run the model over a scan's bytes and report
/// whether the decoder followed it to the last MCU without losing sync.
pub fn follows_scan(file: &[u8], scan_off: usize, len: usize) -> bool {
    let buf = &file[..scan_off + len];
    let mut f = JpegFront::new(&buf[..scan_off], scan_off, len, scan_off);
    let mut st = vec![0i32; JPEG_IN];
    for pos in scan_off..scan_off + len {
        for i in (0..8).rev() {
            let bit = (buf[pos] >> i) as u32 & 1;
            f.inputs(1, 7 - i, &mut st);
            f.update(bit);
        }
        f.locate(pos + 1, buf);
    }
    let s = f.scan.as_ref().unwrap();
    f.mcu >= s.mcux * s.mcuy && f.phase != Phase::Code
}

