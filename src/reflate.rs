//! Reflate: exact re-encoding of deflate streams that zlib did not make.
//!
//! `deflate::find_params` covers streams made by zlib itself. Everything else
//! — zopfli, 7-Zip, Info-ZIP, zlib-ng, Windows' ZIP, PNG optimisers — makes
//! its own choices, and no parameter setting reproduces them. Those streams
//! can still be unpacked, as long as their choices are written down.
//!
//! The choices are cheap to write down because they are mostly predictable
//! from the data. A match is described not by its (length, distance) but
//! relative to what the data offers at that point: how much shorter it is than
//! the longest match available, and which of the candidates that long it took,
//! counted from the most recent. Most encoders take the longest, most recent
//! match, so both numbers are usually zero. Huffman tables are checked against
//! the ones zlib would build from the block's own symbol counts, and stored
//! verbatim only when they differ.
//!
//! The description is a byte string — the "diff" — which travels in the
//! virtual stream ahead of the raw data, where the CM compresses it.

use super::deflate;

const MAX_MATCH: usize = 258;
const WINDOW: usize = 32768;
/// Candidates examined per match. A target beyond this is written out raw.
const MAX_CAND: usize = 4096;
const HASH_BITS: u32 = 18;

const LENGTH_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const EXTRA_LBITS: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const EXTRA_DBITS: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const BL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

#[derive(Clone, Copy)]
enum Tok {
    Lit(u8),
    Match(u16, u16), // length, distance
}

struct Block {
    last: bool,
    btype: u32,
    stored_len: usize,
    pad: u32, // stored blocks: the bits skipped to reach a byte boundary
    toks: Vec<Tok>,
    hdr: Vec<u8>, // dynamic: header bits after BTYPE, LSB-first
    hdr_bits: usize,
    llens: Vec<u8>,
    dlens: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Bits
// ---------------------------------------------------------------------------

struct Bits<'a> {
    d: &'a [u8],
    pos: usize, // bit position
}

impl<'a> Bits<'a> {
    #[inline]
    fn get(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for i in 0..n {
            let byte = *self.d.get(self.pos >> 3)?;
            v |= (((byte >> (self.pos & 7)) & 1) as u32) << i;
            self.pos += 1;
        }
        Some(v)
    }
    /// Decode one symbol of a canonical code given by `lens`.
    fn sym(&mut self, t: &Decode) -> Option<usize> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= self.get(1)? as i32;
            let count = t.counts[len] as i32;
            if code - count < first {
                return t.syms.get((index + (code - first)) as usize).map(|&s| s as usize);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

struct Decode {
    counts: [u16; 16],
    syms: Vec<u16>,
}

fn decoder(lens: &[u8]) -> Option<Decode> {
    let mut counts = [0u16; 16];
    for &l in lens {
        counts[l as usize] += 1;
    }
    counts[0] = 0;
    let mut left: i32 = 1;
    for c in counts.iter().skip(1) {
        left = (left << 1) - *c as i32;
        if left < 0 {
            return None;
        }
    }
    let mut offs = [0u16; 16];
    for len in 1..15 {
        offs[len + 1] = offs[len] + counts[len];
    }
    let mut syms = vec![0u16; lens.len()];
    for (s, &l) in lens.iter().enumerate() {
        if l != 0 {
            syms[offs[l as usize] as usize] = s as u16;
            offs[l as usize] += 1;
        }
    }
    Some(Decode { counts, syms })
}

/// Canonical codes (bit-reversed for LSB-first output) for `lens`.
fn codes(lens: &[u8]) -> Vec<u16> {
    let mut bl = [0u16; 16];
    for &l in lens {
        bl[l as usize] += 1;
    }
    bl[0] = 0;
    let mut next = [0u32; 16];
    let mut c = 0u32;
    for b in 1..16 {
        c = (c + bl[b - 1] as u32) << 1;
        next[b] = c;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                return 0;
            }
            let v = next[l as usize];
            next[l as usize] += 1;
            let mut r = 0u32;
            for i in 0..l {
                r |= ((v >> i) & 1) << (l - 1 - i);
            }
            r as u16
        })
        .collect()
}

fn fixed_lens() -> (Vec<u8>, Vec<u8>) {
    let mut l = vec![0u8; 288];
    l[..144].fill(8);
    l[144..256].fill(9);
    l[256..280].fill(7);
    l[280..].fill(8);
    (l, vec![5u8; 30])
}

/// Read a dynamic header (after BTYPE): returns the code lengths.
fn read_dyn_header(b: &mut Bits) -> Option<(Vec<u8>, Vec<u8>)> {
    let hlit = b.get(5)? as usize + 257;
    let hdist = b.get(5)? as usize + 1;
    let hclen = b.get(4)? as usize + 4;
    let mut cl = [0u8; 19];
    for &o in BL_ORDER.iter().take(hclen) {
        cl[o] = b.get(3)? as u8;
    }
    let clt = decoder(&cl)?;
    let mut lens = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        let sym = b.sym(&clt)?;
        let (val, rep) = match sym {
            0..=15 => (sym as u8, 1),
            16 => (*lens.get(i.checked_sub(1)?)?, 3 + b.get(2)? as usize),
            17 => (0, 3 + b.get(3)? as usize),
            18 => (0, 11 + b.get(7)? as usize),
            _ => return None,
        };
        if i + rep > hlit + hdist {
            return None;
        }
        lens[i..i + rep].fill(val);
        i += rep;
    }
    let mut ll = lens[..hlit].to_vec();
    ll.resize(288, 0);
    let mut dl = lens[hlit..].to_vec();
    dl.resize(30, 0);
    Some((ll, dl))
}

fn copy_bits(d: &[u8], from: usize, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n.div_ceil(8)];
    for i in 0..n {
        let bit = (d[(from + i) >> 3] >> ((from + i) & 7)) & 1;
        out[i >> 3] |= bit << (i & 7);
    }
    out
}

/// Parse a raw deflate stream into blocks and tokens. Returns the blocks, the
/// inflated data, and the stream's length in bytes.
fn parse(d: &[u8], limit: usize) -> Option<(Vec<Block>, Vec<u8>, usize, u32)> {
    let mut b = Bits { d, pos: 0 };
    let mut blocks = Vec::new();
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = b.get(1)? == 1;
        let btype = b.get(2)?;
        let mut blk = Block {
            last,
            btype,
            stored_len: 0,
            pad: 0,
            toks: Vec::new(),
            hdr: Vec::new(),
            hdr_bits: 0,
            llens: Vec::new(),
            dlens: Vec::new(),
        };
        match btype {
            0 => {
                let n = (8 - (b.pos & 7)) & 7;
                blk.pad = b.get(n as u32)?;
                let len = b.get(16)? as usize;
                let nlen = b.get(16)? as usize;
                if len != !nlen & 0xffff {
                    return None;
                }
                let at = b.pos >> 3;
                out.extend_from_slice(d.get(at..at + len)?);
                b.pos += len * 8;
                blk.stored_len = len;
            }
            1 | 2 => {
                let (ll, dl) = if btype == 1 {
                    fixed_lens()
                } else {
                    let st = b.pos;
                    let (ll, dl) = read_dyn_header(&mut b)?;
                    blk.hdr_bits = b.pos - st;
                    blk.hdr = copy_bits(d, st, blk.hdr_bits);
                    (ll, dl)
                };
                let (lt, dt) = (decoder(&ll)?, decoder(&dl)?);
                loop {
                    let sym = b.sym(&lt)?;
                    if sym < 256 {
                        out.push(sym as u8);
                        blk.toks.push(Tok::Lit(sym as u8));
                    } else if sym == 256 {
                        break;
                    } else {
                        let k = sym - 257;
                        if k >= 29 {
                            return None;
                        }
                        let len = LENGTH_BASE[k] as usize + b.get(EXTRA_LBITS[k] as u32)? as usize;
                        let dk = b.sym(&dt)?;
                        if dk >= 30 {
                            return None;
                        }
                        let dist = DIST_BASE[dk] as usize + b.get(EXTRA_DBITS[dk] as u32)? as usize;
                        if dist > out.len() {
                            return None;
                        }
                        let from = out.len() - dist;
                        for j in 0..len {
                            let v = out[from + j];
                            out.push(v);
                        }
                        blk.toks.push(Tok::Match(len as u16, dist as u16));
                    }
                    if out.len() > limit {
                        return None;
                    }
                }
                blk.llens = ll;
                blk.dlens = dl;
            }
            _ => return None,
        }
        blocks.push(blk);
        if last {
            break;
        }
    }
    // the bits left over in the final byte
    let tail = (8 - (b.pos & 7)) & 7;
    let pad = b.get(tail as u32)?;
    Some((blocks, out, b.pos >> 3, pad))
}

// ---------------------------------------------------------------------------
// Match candidates: the shared view of "what the data offers"
// ---------------------------------------------------------------------------

struct Chains {
    head: Vec<u32>, // hash -> 1 + most recent position
    prev: Vec<u32>, // position -> 1 + previous position with that hash
    next_insert: usize,
}

#[inline]
fn h3(d: &[u8], i: usize) -> usize {
    let v = (d[i] as u32) | (d[i + 1] as u32) << 8 | (d[i + 2] as u32) << 16;
    (v.wrapping_mul(0x9e37_79b1) >> (32 - HASH_BITS)) as usize
}

impl Chains {
    fn new(n: usize) -> Self {
        Self { head: vec![0; 1 << HASH_BITS], prev: vec![0; n], next_insert: 0 }
    }

    /// Make every position before `upto` a candidate.
    fn insert_to(&mut self, d: &[u8], upto: usize) {
        while self.next_insert < upto {
            let i = self.next_insert;
            if i + 3 <= d.len() {
                let h = h3(d, i);
                self.prev[i] = self.head[h];
                self.head[h] = i as u32 + 1;
            }
            self.next_insert += 1;
        }
    }

    /// Candidates for a match at `pos`, most recent first: (distance, length
    /// available), cut short once one reaches the longest length possible.
    fn candidates(&self, d: &[u8], pos: usize, out: &mut Vec<(usize, usize)>) {
        out.clear();
        if pos + 3 > d.len() {
            return;
        }
        let cap = MAX_MATCH.min(d.len() - pos);
        let mut c = self.head[h3(d, pos)];
        let mut steps = 0;
        while c != 0 && steps < MAX_CAND {
            let q = c as usize - 1;
            let dist = pos - q;
            if dist > WINDOW {
                break;
            }
            steps += 1;
            if d[q] == d[pos] && d[q + 1] == d[pos + 1] && d[q + 2] == d[pos + 2] {
                let mut l = 3;
                while l < cap && d[q + l] == d[pos + l] {
                    l += 1;
                }
                out.push((dist, l));
                if l == cap {
                    break;
                }
            }
            c = self.prev[q];
        }
    }
}

fn put_var(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_var(d: &[u8], p: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *d.get(*p)?;
        *p += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

// Token records. Every token is first compared with what a zlib-style lazy
// matcher would emit at that point, given the data: most encoders agree with
// it most of the time, and agreement is one zero byte.
const T_AS_PREDICTED: u8 = 0;
const T_LIT: u8 = 1; // a literal where a match was predicted
const T_MATCH: u8 = 2; // T_MATCH + (longest - len), then the rank byte
const T_RAW: u8 = 255; // match written out: varint len-3, varint dist
const TOO_FAR: usize = 4096;

/// The view at one position: the longest match available, its most recent
/// distance, and all candidates.
struct Here {
    maxall: usize,
    dist: usize,
    cand: Vec<(usize, usize)>,
}

/// Match-or-literal prediction, walked in step by both sides. It starts out
/// as zlib's lazy rule and then learns the encoder's actual policy: per
/// situation (match length here, how much longer the next position's is, how
/// far back), whichever decision the encoder has made more often.
struct Lazy {
    ch: Chains,
    next: Option<(usize, usize)>, // (position, maxall) already computed for p+1
    policy: Vec<[u32; 2]>,        // situation -> (literals, matches) seen
    sit: Option<usize>,           // situation of the token being decided
}

const SITUATIONS: usize = 11 * 11 * 3;

impl Lazy {
    fn new(n: usize) -> Self {
        Self { ch: Chains::new(n), next: None, policy: vec![[0; 2]; SITUATIONS], sit: None }
    }

    /// Record what the encoder did at the situation last predicted.
    fn learn(&mut self, took_match: bool) {
        if let Some(k) = self.sit.take() {
            let c = &mut self.policy[k][took_match as usize];
            *c = c.saturating_add(1);
        }
    }

    fn here(&mut self, d: &[u8], pos: usize) -> Here {
        self.ch.insert_to(d, pos);
        let mut cand = Vec::new();
        self.ch.candidates(d, pos, &mut cand);
        let (mut maxall, mut dist) = (0, 0);
        for &(dd, l) in &cand {
            if l > maxall {
                maxall = l;
                dist = dd;
            }
        }
        Here { maxall, dist, cand }
    }

    /// What a lazy matcher emits at `pos`: Some(len, dist) or None (literal).
    fn predict(&mut self, d: &[u8], pos: usize, h: &Here) -> Option<(usize, usize)> {
        self.sit = None;
        if h.maxall < 3 {
            return None;
        }
        let m1 = match self.next {
            Some((p, m)) if p == pos + 1 => m,
            _ => {
                let m = if pos + 1 < d.len() { self.here(d, pos + 1).maxall } else { 0 };
                self.next = Some((pos + 1, m));
                m
            }
        };
        let far = if h.dist <= 256 { 0 } else if h.dist <= TOO_FAR { 1 } else { 2 };
        let k = ((h.maxall.min(10) * 11) + (m1.saturating_sub(h.maxall)).min(10)) * 3 + far;
        self.sit = Some(k);
        let [lits, matches] = self.policy[k];
        // zlib's rule until the encoder shows otherwise: a longer match one
        // byte on wins (this byte goes out as a literal), and a 3-byte match
        // too far back isn't worth its distance code
        let zlib_match = !(m1 > h.maxall || (h.maxall == 3 && h.dist > TOO_FAR));
        let take = if lits == matches { zlib_match } else { matches > lits };
        take.then_some((h.maxall, h.dist))
    }
}

/// Symbol frequencies of a block as zlib counts them (END_BLOCK included).
fn freqs(toks: &[Tok]) -> (Vec<u16>, Vec<u16>) {
    let mut lf = vec![0u16; 286];
    let mut df = vec![0u16; 30];
    for t in toks {
        match *t {
            Tok::Lit(c) => lf[c as usize] = lf[c as usize].wrapping_add(1),
            Tok::Match(len, dist) => {
                let lk = LENGTH_BASE.iter().rposition(|&b| b as usize <= len as usize).unwrap();
                let lk = if len == 258 { 28 } else { lk.min(27) };
                lf[257 + lk] = lf[257 + lk].wrapping_add(1);
                let dk = DIST_BASE.iter().rposition(|&b| b <= dist).unwrap();
                df[dk] = df[dk].wrapping_add(1);
            }
        }
    }
    lf[256] = 1;
    (lf, df)
}

/// Analyse a raw deflate stream. Returns (inflated data, diff, bytes consumed)
/// such that `rebuild(data, diff)` reproduces the stream exactly — verified
/// before returning.
pub fn analyze(comp: &[u8], limit: usize) -> Option<(Vec<u8>, Vec<u8>, usize)> {
    let (blocks, data, consumed, tail_pad) = parse(comp, limit)?;
    let mut diff = Vec::new();
    put_var(&mut diff, blocks.len() as u64);
    let mut lz = Lazy::new(data.len());
    let mut pos = 0usize;
    for blk in &blocks {
        let mut hdr_raw = false;
        if blk.btype == 2 {
            let (lf, df) = freqs(&blk.toks);
            let (zh, zbits, _, _) = deflate::zlib_dynamic_header(&lf, &df);
            hdr_raw = zbits != blk.hdr_bits || zh != blk.hdr;
        }
        diff.push(blk.btype as u8 | (blk.last as u8) << 2 | (hdr_raw as u8) << 3);
        if blk.btype == 0 {
            put_var(&mut diff, blk.stored_len as u64);
            diff.push(blk.pad as u8);
            pos += blk.stored_len;
            continue;
        }
        put_var(&mut diff, blk.toks.len() as u64);
        if hdr_raw {
            put_var(&mut diff, blk.hdr_bits as u64);
            diff.extend_from_slice(&blk.hdr);
        }
        // agreements are counted, not written: a run length, then the exception
        let mut run = 0u64;
        for t in &blk.toks {
            let h = lz.here(&data, pos);
            let pred = lz.predict(&data, pos, &h);
            let agrees = match *t {
                Tok::Lit(_) => pred.is_none(),
                Tok::Match(len, dist) => pred == Some((len as usize, dist as usize)),
            };
            if agrees {
                run += 1;
            } else {
                put_var(&mut diff, run);
                run = 0;
            }
            match *t {
                Tok::Lit(_) => {
                    if !agrees {
                        diff.push(T_LIT);
                    }
                    lz.learn(false);
                    pos += 1;
                }
                Tok::Match(len, dist) => {
                    let (len, dist) = (len as usize, dist as usize);
                    lz.learn(true);
                    if agrees {
                    } else {
                        // rank among candidates at least this long, most recent first
                        let rank = h.cand.iter().filter(|c| c.1 >= len).position(|c| c.0 == dist);
                        match rank {
                            Some(r) if h.maxall >= len && h.maxall - len < 250 && r < 255 => {
                                diff.push(T_MATCH + (h.maxall - len) as u8);
                                diff.push(r as u8);
                            }
                            _ => {
                                diff.push(T_RAW);
                                put_var(&mut diff, len as u64 - 3);
                                put_var(&mut diff, dist as u64);
                            }
                        }
                    }
                    pos += len;
                }
            }
        }
        if run > 0 {
            put_var(&mut diff, run);
        }
    }
    diff.push(tail_pad as u8);
    let back = rebuild(&data, &diff)?;
    (back == comp[..consumed]).then_some((data, diff, consumed))
}

struct Out {
    v: Vec<u8>,
    buf: u64,
    n: u32,
}

impl Out {
    #[inline]
    fn send(&mut self, value: u32, len: u32) {
        self.buf |= (value as u64) << self.n;
        self.n += len;
        while self.n >= 8 {
            self.v.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }
    fn send_bits(&mut self, bytes: &[u8], nbits: usize) {
        for i in 0..nbits {
            self.send(((bytes[i >> 3] >> (i & 7)) & 1) as u32, 1);
        }
    }
}

/// Re-encode the deflate stream described by `diff` over `data`.
pub fn rebuild(data: &[u8], diff: &[u8]) -> Option<Vec<u8>> {
    let mut p = 0usize;
    let nblocks = get_var(diff, &mut p)? as usize;
    if nblocks > diff.len() {
        return None;
    }
    let mut o = Out { v: Vec::with_capacity(data.len() / 2), buf: 0, n: 0 };
    let mut lz = Lazy::new(data.len());
    let mut pos = 0usize;
    for _ in 0..nblocks {
        let f = *diff.get(p)?;
        p += 1;
        let (btype, last, hdr_raw) = ((f & 3) as u32, f >> 2 & 1, f >> 3 & 1 == 1);
        o.send(last as u32, 1);
        o.send(btype, 2);
        if btype == 0 {
            let len = get_var(diff, &mut p)? as usize;
            let pad = *diff.get(p)? as u32;
            p += 1;
            let n = (8 - o.n) & 7;
            o.send(pad, n);
            o.send(len as u32 & 0xffff, 16);
            o.send(!len as u32 & 0xffff, 16);
            for &byte in data.get(pos..pos.checked_add(len)?)? {
                o.send(byte as u32, 8);
            }
            pos += len;
            continue;
        }
        if btype == 3 {
            return None;
        }
        let ntok = get_var(diff, &mut p)? as usize;
        if ntok > data.len() + 1 {
            return None;
        }
        let hdr: Option<(Vec<u8>, usize)> = if hdr_raw {
            let nbits = get_var(diff, &mut p)? as usize;
            let bytes = diff.get(p..p.checked_add(nbits.div_ceil(8))?)?.to_vec();
            p += nbits.div_ceil(8);
            Some((bytes, nbits))
        } else {
            None
        };
        // tokens first: a predicted header depends on their counts
        let mut toks = Vec::with_capacity(ntok);
        // [run][exception] pairs; a spent run means the next token is the exception
        let mut run = 0u64;
        let mut want_run = true;
        for _ in 0..ntok {
            if pos >= data.len() {
                return None;
            }
            let h = lz.here(data, pos);
            let pred = lz.predict(data, pos, &h);
            if want_run {
                run = get_var(diff, &mut p)?;
                want_run = false;
            }
            let b = if run > 0 {
                run -= 1;
                T_AS_PREDICTED
            } else {
                // the run is spent: this token is the exception
                want_run = true;
                let b = *diff.get(p)?;
                p += 1;
                if b == T_AS_PREDICTED {
                    return None;
                }
                b
            };

            let lit = match b {
                T_AS_PREDICTED => pred.is_none(),
                T_LIT => true,
                _ => false,
            };
            lz.learn(!lit);
            if lit {
                toks.push(Tok::Lit(data[pos]));
                pos += 1;
                continue;
            }
            let (len, dist) = if b == T_AS_PREDICTED {
                pred?
            } else if b == T_RAW {
                let len = get_var(diff, &mut p)? as usize + 3;
                let dist = get_var(diff, &mut p)? as usize;
                (len, dist)
            } else {
                let r = *diff.get(p)? as usize;
                p += 1;
                let len = h.maxall.checked_sub((b - T_MATCH) as usize)?;
                let dist = h.cand.iter().filter(|c| c.1 >= len).nth(r)?.0;
                (len, dist)
            };
            if !(3..=MAX_MATCH).contains(&len) || dist == 0 || dist > pos || dist > WINDOW {
                return None;
            }
            toks.push(Tok::Match(len as u16, dist as u16));
            pos += len;
        }
        let (ll, dl) = match (btype, hdr) {
            (1, _) => fixed_lens(),
            (_, Some((bytes, nbits))) => {
                o.send_bits(&bytes, nbits);
                let mut b = Bits { d: &bytes, pos: 0 };
                read_dyn_header(&mut b)?
            }
            _ => {
                let (lf, df) = freqs(&toks);
                let (zh, zbits, mut ll, mut dl) = deflate::zlib_dynamic_header(&lf, &df);
                o.send_bits(&zh, zbits);
                ll.resize(288, 0);
                dl.resize(30, 0);
                (ll, dl)
            }
        };
        let (lc, dc) = (codes(&ll), codes(&dl));
        for t in &toks {
            match *t {
                Tok::Lit(c) => o.send(lc[c as usize] as u32, ll[c as usize] as u32),
                Tok::Match(len, dist) => {
                    let len = len as usize;
                    let k = if len == 258 { 28 } else { LENGTH_BASE.iter().rposition(|&b| b as usize <= len)?.min(27) };
                    if ll[257 + k] == 0 {
                        return None;
                    }
                    o.send(lc[257 + k] as u32, ll[257 + k] as u32);
                    o.send((len - LENGTH_BASE[k] as usize) as u32, EXTRA_LBITS[k] as u32);
                    let dk = DIST_BASE.iter().rposition(|&b| b <= dist)?;
                    if dl[dk] == 0 {
                        return None;
                    }
                    o.send(dc[dk] as u32, dl[dk] as u32);
                    o.send((dist - DIST_BASE[dk]) as u32, EXTRA_DBITS[dk] as u32);
                }
            }
        }
        if ll[256] == 0 {
            return None;
        }
        o.send(lc[256] as u32, ll[256] as u32);
    }
    let tail = *diff.get(p)? as u32;
    let n = (8 - o.n) & 7;
    o.send(tail, n);
    (pos == data.len()).then_some(o.v)
}

/// Analysis aid: describe a stream's diff composition.
pub fn stats(comp: &[u8]) -> Option<String> {
    let (blocks, data, _, _) = parse(comp, usize::MAX)?;
    let (_, diff, _) = analyze(comp, usize::MAX)?;
    let mut lz = Lazy::new(data.len());
    let mut c = [0usize; 8]; // lit ok, lit-but-match-predicted, match ok, match-but-lit-predicted, match other: shorter, other rank, raw
    let mut hdr_raw = 0;
    let mut short_hist = [0usize; 8];
    let mut mm = std::collections::BTreeMap::new(); // (m0, m1-m0, took longest) for matches taken against the lazy rule
    let mut ml = std::collections::BTreeMap::new(); // (m0, m1-m0) for literals the lazy rule explained
    let mut pos = 0;
    for blk in &blocks {
        if blk.btype == 2 {
            let (lf, df) = freqs(&blk.toks);
            let (zh, zbits, _, _) = deflate::zlib_dynamic_header(&lf, &df);
            hdr_raw += (zbits != blk.hdr_bits || zh != blk.hdr) as usize;
        }
        if blk.btype == 0 {
            pos += blk.stored_len;
            continue;
        }
        for t in &blk.toks {
            let h = lz.here(&data, pos);
            let pred = lz.predict(&data, pos, &h);
            match *t {
                Tok::Lit(_) => {
                    lz.learn(false);
                    c[if pred.is_none() { 0 } else { 1 }] += 1;
                    if pred.is_none() && h.maxall >= 3 {
                        let m1 = lz.next.map(|x| x.1).unwrap_or(0);
                        *ml.entry((h.maxall.min(9), m1.saturating_sub(h.maxall).min(9))).or_insert(0usize) += 1;
                    }
                    pos += 1;
                }
                Tok::Match(len, dist) => {
                    let (len, dist) = (len as usize, dist as usize);
                    lz.learn(true);
                    if pred == Some((len, dist)) {
                        c[2] += 1;
                    } else if pred.is_none() {
                        c[3] += 1;
                        let m1 = lz.next.map(|x| x.1).unwrap_or(0);
                        *mm.entry((h.maxall.min(9), (m1 - h.maxall).min(9), len == h.maxall)).or_insert(0usize) += 1;
                    } else if len < h.maxall {
                        c[4] += 1;
                        short_hist[(h.maxall - len).min(7)] += 1;
                    } else {
                        c[5] += 1;
                    }
                    pos += len;
                }
            }
        }
    }
    Some(format!(
        "blocks {} (hdr raw {})  data {}  diff {}\nlit ok {}  lit-but-match-predicted {}\nmatch ok {}  match-but-lit-predicted {}  shorter {} {:?}  other dist {}",
        blocks.len(), hdr_raw, data.len(), diff.len(), c[0], c[1], c[2], c[3], c[4], short_hist, c[5]
    ) + &format!("\nagainst-lazy matches (m0, m1-m0, longest): {:?}\nlazy literals (m0, m1-m0): {:?}", mm, ml))
}
