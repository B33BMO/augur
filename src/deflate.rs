//! Deflate, both ways.
//!
//! Most of the world's compressed bytes are deflate: PNG, ZIP, gzip, Office
//! documents, PDF streams, JAR/APK. To a context mixer those bytes are noise —
//! the redundancy has already been squeezed out, by a much weaker model. The
//! way to compress them well is to *undo* the deflate, model the raw data, and
//! on decompression redo the deflate — which is only lossless if the redo
//! reproduces the original stream bit for bit.
//!
//! That is possible because almost every deflate stream in the wild was made by
//! zlib, and zlib is deterministic: the same input and the same (level,
//! memLevel, windowBits, strategy) always produce the same bits. So this module
//! holds an inflater, and a clean-room reimplementation of zlib's compressor —
//! its hash chains, lazy-match heuristics, window sliding, block-split points
//! and Huffman tree construction, tie-breaking included — precise enough that
//! its output can be compared byte for byte with the original. When it matches,
//! the stream is stored as its parameters plus the raw data. When it doesn't
//! (another encoder made it), the stream is left alone.

// ---------------------------------------------------------------------------
// Shared tables
// ---------------------------------------------------------------------------

const LENGTH_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const EXTRA_LBITS: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const EXTRA_DBITS: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; 19] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

// ---------------------------------------------------------------------------
// Inflate
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    d: &'a [u8],
    pos: usize, // next byte
    bits: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, pos: 0, bits: 0, n: 0 }
    }
    #[inline]
    fn need(&mut self, k: u32) -> Option<()> {
        while self.n < k {
            let b = *self.d.get(self.pos)?;
            self.pos += 1;
            self.bits |= (b as u64) << self.n;
            self.n += 8;
        }
        Some(())
    }
    #[inline]
    fn get(&mut self, k: u32) -> Option<u32> {
        if k == 0 {
            return Some(0);
        }
        self.need(k)?;
        let v = (self.bits & ((1u64 << k) - 1)) as u32;
        self.bits >>= k;
        self.n -= k;
        Some(v)
    }
    fn align(&mut self) {
        let r = self.n % 8;
        self.bits >>= r;
        self.n -= r;
    }
    /// Bytes consumed, counting a partially used byte as consumed.
    fn consumed(&self) -> usize {
        self.pos - (self.n / 8) as usize
    }
}

/// Canonical Huffman decoding table: (symbol, length) per `fast`-bit prefix,
/// falling back to a bit-by-bit walk for longer codes.
struct Huff {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huff {
    fn new(lens: &[u8]) -> Option<Huff> {
        let mut counts = [0u16; 16];
        for &l in lens {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        // reject over-subscribed codes; incomplete ones are legal (one-code trees)
        let mut left: i32 = 1;
        for len in 1..16 {
            left <<= 1;
            left -= counts[len] as i32;
            if left < 0 {
                return None;
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + counts[len];
        }
        let mut symbols = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Some(Huff { counts, symbols })
    }

    #[inline]
    fn decode(&self, br: &mut BitReader) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= br.get(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return self.symbols.get((index + (code - first)) as usize).copied();
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

fn fixed_huff() -> (Huff, Huff) {
    let mut l = [0u8; 288];
    l[..144].fill(8);
    l[144..256].fill(9);
    l[256..280].fill(7);
    l[280..].fill(8);
    (Huff::new(&l).unwrap(), Huff::new(&[5u8; 30]).unwrap())
}

/// Result of inflating a raw deflate stream.
pub struct Inflated {
    pub data: Vec<u8>,
    /// Compressed bytes consumed, up to and including the final block's last
    /// partially-used byte.
    pub consumed: usize,
}

/// Inflate a raw deflate stream. `limit` caps the output, so a hostile stream
/// cannot balloon memory. Returns None on any malformation.
pub fn inflate(d: &[u8], limit: usize) -> Option<Inflated> {
    inflate_impl(d, limit, None)
}

/// One block of a deflate stream, for analysis.
#[derive(Debug, Clone)]
pub struct BlockInfo {
    pub btype: u32,
    pub out_start: usize,
    pub out_len: usize,
    pub symbols: usize, // literals + matches, end-of-block excluded
    pub matches: usize,
}

pub fn inflate_blocks(d: &[u8]) -> Option<Vec<BlockInfo>> {
    let mut v = Vec::new();
    inflate_impl(d, usize::MAX, Some(&mut v))?;
    Some(v)
}

fn inflate_impl(d: &[u8], limit: usize, mut log: Option<&mut Vec<BlockInfo>>) -> Option<Inflated> {
    let mut br = BitReader::new(d);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = br.get(1)?;
        let kind = br.get(2)?;
        let out_start = out.len();
        let (mut nsym, mut nmatch) = (0usize, 0usize);
        match kind {
            0 => {
                br.align();
                let len = br.get(16)?;
                let nlen = br.get(16)?;
                if len != !nlen & 0xffff {
                    return None;
                }
                // the reader may hold whole bytes already; take them first
                for _ in 0..len {
                    out.push(br.get(8)? as u8);
                }
            }
            1 | 2 => {
                let (lit, dist) = if kind == 1 {
                    fixed_huff()
                } else {
                    let hlit = br.get(5)? as usize + 257;
                    let hdist = br.get(5)? as usize + 1;
                    let hclen = br.get(4)? as usize + 4;
                    let mut cl = [0u8; 19];
                    for &o in BL_ORDER.iter().take(hclen) {
                        cl[o] = br.get(3)? as u8;
                    }
                    let clh = Huff::new(&cl)?;
                    let mut lens = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < hlit + hdist {
                        let sym = clh.decode(&mut br)?;
                        let (val, rep) = match sym {
                            0..=15 => (sym as u8, 1),
                            16 => (*lens.get(i.checked_sub(1)?)?, 3 + br.get(2)? as usize),
                            17 => (0, 3 + br.get(3)? as usize),
                            18 => (0, 11 + br.get(7)? as usize),
                            _ => return None,
                        };
                        if i + rep > hlit + hdist {
                            return None;
                        }
                        lens[i..i + rep].fill(val);
                        i += rep;
                    }
                    if lens[256] == 0 {
                        return None;
                    }
                    (Huff::new(&lens[..hlit])?, Huff::new(&lens[hlit..])?)
                };
                loop {
                    let sym = lit.decode(&mut br)? as usize;
                    if sym < 256 {
                        out.push(sym as u8);
                        nsym += 1;
                    } else if sym == 256 {
                        break;
                    } else {
                        nsym += 1;
                        nmatch += 1;
                        let k = sym - 257;
                        if k >= 29 {
                            return None;
                        }
                        let len = LENGTH_BASE[k] as usize + br.get(EXTRA_LBITS[k] as u32)? as usize;
                        let dk = dist.decode(&mut br)? as usize;
                        if dk >= 30 {
                            return None;
                        }
                        let dd = DIST_BASE[dk] as usize + br.get(EXTRA_DBITS[dk] as u32)? as usize;
                        if dd > out.len() {
                            return None;
                        }
                        let from = out.len() - dd;
                        for j in 0..len {
                            let b = out[from + j];
                            out.push(b);
                        }
                    }
                    if out.len() > limit {
                        return None;
                    }
                }
            }
            _ => return None,
        }
        if out.len() > limit {
            return None;
        }
        if let Some(l) = log.as_deref_mut() {
            l.push(BlockInfo { btype: kind, out_start, out_len: out.len() - out_start, symbols: nsym, matches: nmatch });
        }
        if last == 1 {
            break;
        }
    }
    Some(Inflated { data: out, consumed: br.consumed() })
}

// ---------------------------------------------------------------------------
// zlib-exact deflate
// ---------------------------------------------------------------------------

/// zlib's compression strategies.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Strategy {
    Default = 0,
    Filtered = 1,
    HuffmanOnly = 2,
    Rle = 3,
    Fixed = 4,
}

impl Strategy {
    pub fn from_u8(v: u8) -> Option<Strategy> {
        Some(match v {
            0 => Strategy::Default,
            1 => Strategy::Filtered,
            2 => Strategy::HuffmanOnly,
            3 => Strategy::Rle,
            4 => Strategy::Fixed,
            _ => return None,
        })
    }
}

/// Everything that determines zlib's output for a given input.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Params {
    pub level: u8,     // 1..=9
    pub mem_level: u8, // 1..=9
    pub wbits: u8,     // 9..=15 (zlib treats 8 as 9)
    pub strategy: Strategy,
}

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const TOO_FAR: usize = 4096;
const L_CODES: usize = 286;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;

/// (good_length, max_lazy, nice_length, max_chain) per level, from zlib.
const CONFIG: [(usize, usize, usize, usize); 10] = [
    (0, 0, 0, 0),
    (4, 4, 8, 4),
    (4, 5, 16, 8),
    (4, 6, 32, 32),
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

struct Tables {
    length_code: [u8; 256],
    dist_code: [u8; 512],
    base_length: [u8; 29],
    base_dist: [u16; 30],
    static_ltree: [(u16, u8); L_CODES + 2], // (code, len)
    static_dtree: [(u16, u8); D_CODES],
}

fn bi_reverse(mut code: u32, mut len: u32) -> u32 {
    let mut res = 0;
    loop {
        res |= code & 1;
        code >>= 1;
        res <<= 1;
        len -= 1;
        if len == 0 {
            break;
        }
    }
    res >> 1
}

/// Assign canonical codes to `len`, zlib's gen_codes.
fn gen_codes(len: &[u8], max_code: usize, bl_count: &[u16; MAX_BITS + 1], code: &mut [u16]) {
    let mut next = [0u32; MAX_BITS + 1];
    let mut c = 0u32;
    for bits in 1..=MAX_BITS {
        c = (c + bl_count[bits - 1] as u32) << 1;
        next[bits] = c;
    }
    for n in 0..=max_code {
        let l = len[n] as usize;
        if l == 0 {
            continue;
        }
        code[n] = bi_reverse(next[l], l as u32) as u16;
        next[l] += 1;
    }
}

fn tables() -> &'static Tables {
    use std::sync::OnceLock;
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            length_code: [0; 256],
            dist_code: [0; 512],
            base_length: [0; 29],
            base_dist: [0; 30],
            static_ltree: [(0, 0); L_CODES + 2],
            static_dtree: [(0, 0); D_CODES],
        };
        let mut length = 0usize;
        let mut code = 0usize;
        while code < 28 {
            t.base_length[code] = length as u8;
            for _ in 0..(1 << EXTRA_LBITS[code]) {
                t.length_code[length] = code as u8;
                length += 1;
            }
            code += 1;
        }
        // length 258 (255 after the bias) gets the dedicated code 28
        t.length_code[length - 1] = code as u8;
        t.base_length[28] = 0; // unused: code 28 has no extra bits
        let mut dist = 0usize;
        code = 0;
        while code < 16 {
            t.base_dist[code] = dist as u16;
            for _ in 0..(1 << EXTRA_DBITS[code]) {
                t.dist_code[dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        dist >>= 7;
        while code < D_CODES {
            t.base_dist[code] = (dist << 7) as u16;
            for _ in 0..(1 << (EXTRA_DBITS[code] - 7)) {
                t.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
            code += 1;
        }
        let mut lens = [0u8; L_CODES + 2];
        let mut bl_count = [0u16; MAX_BITS + 1];
        for (n, l) in lens.iter_mut().enumerate() {
            *l = match n {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
            bl_count[*l as usize] += 1;
        }
        let mut codes = [0u16; L_CODES + 2];
        gen_codes(&lens, L_CODES + 1, &bl_count, &mut codes);
        for n in 0..L_CODES + 2 {
            t.static_ltree[n] = (codes[n], lens[n]);
        }
        for n in 0..D_CODES {
            t.static_dtree[n] = (bi_reverse(n as u32, 5) as u16, 5);
        }
        t
    })
}

#[inline]
fn d_code(t: &Tables, dist: usize) -> usize {
    if dist < 256 {
        t.dist_code[dist] as usize
    } else {
        t.dist_code[256 + (dist >> 7)] as usize
    }
}

/// One Huffman tree under construction, zlib's ct_data split into arrays.
struct Tree {
    freq: Vec<u16>,
    len: Vec<u8>,
    code: Vec<u16>,
    dad: Vec<u16>,
    max_code: usize,
}

impl Tree {
    fn new(n: usize) -> Self {
        Self { freq: vec![0; n], len: vec![0; n], code: vec![0; n], dad: vec![0; n], max_code: 0 }
    }
}

struct BitWriter {
    out: Vec<u8>,
    buf: u64,
    n: u32,
}

impl BitWriter {
    #[inline]
    fn send(&mut self, value: u32, len: u32) {
        self.buf |= (value as u64) << self.n;
        self.n += len;
        while self.n >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }
    fn windup(&mut self) {
        if self.n > 0 {
            self.out.push(self.buf as u8);
        }
        self.buf = 0;
        self.n = 0;
    }
}

struct Deflater<'a> {
    input: &'a [u8],
    in_pos: usize,
    p: Params,
    w_size: usize,
    w_mask: usize,
    window: Vec<u8>,
    window_size: usize,
    prev: Vec<u16>,
    head: Vec<u16>,
    ins_h: usize,
    hash_mask: usize,
    hash_shift: u32,
    block_start: isize,
    match_length: usize,
    prev_match: usize,
    match_available: bool,
    strstart: usize,
    match_start: usize,
    lookahead: usize,
    prev_length: usize,
    max_chain: usize,
    max_lazy: usize,
    good_match: usize,
    nice_match: usize,
    insert: usize,
    // symbol buffer: (dist, lc) as zlib's sym_buf holds them
    syms: Vec<(u16, u8)>,
    sym_limit: usize,
    ltree: Tree,
    dtree: Tree,
    bltree: Tree,
    heap: [usize; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
    bl_count: [u16; MAX_BITS + 1],
    opt_len: u64,
    static_len: u64,
    bw: BitWriter,
    // verification mode: stop at the first block that disagrees with this
    expect: Option<&'a [u8]>,
    diverged: bool,
}

impl<'a> Deflater<'a> {
    fn new(input: &'a [u8], p: Params) -> Self {
        let wbits = p.wbits.max(9) as usize;
        let w_size = 1usize << wbits;
        let hash_bits = p.mem_level as usize + 7;
        let hash_size = 1usize << hash_bits;
        let lit_bufsize = 1usize << (p.mem_level as usize + 6);
        let (good, lazy, nice, chain) = CONFIG[p.level as usize];
        let mut d = Deflater {
            input,
            in_pos: 0,
            p,
            w_size,
            w_mask: w_size - 1,
            window: vec![0; 2 * w_size],
            window_size: 2 * w_size,
            prev: vec![0; w_size],
            head: vec![0; hash_size],
            ins_h: 0,
            hash_mask: hash_size - 1,
            hash_shift: ((hash_bits + MIN_MATCH - 1) / MIN_MATCH) as u32,
            block_start: 0,
            match_length: MIN_MATCH - 1,
            prev_match: 0,
            match_available: false,
            strstart: 0,
            match_start: 0,
            lookahead: 0,
            prev_length: MIN_MATCH - 1,
            max_chain: chain,
            max_lazy: lazy,
            good_match: good,
            nice_match: nice,
            insert: 0,
            syms: Vec::with_capacity(lit_bufsize),
            sym_limit: lit_bufsize - 1,
            ltree: Tree::new(HEAP_SIZE),
            dtree: Tree::new(2 * D_CODES + 1),
            bltree: Tree::new(2 * BL_CODES + 1),
            heap: [0; HEAP_SIZE],
            heap_len: 0,
            heap_max: 0,
            depth: [0; HEAP_SIZE],
            bl_count: [0; MAX_BITS + 1],
            opt_len: 0,
            static_len: 0,
            bw: BitWriter { out: Vec::with_capacity(input.len() / 2 + 64), buf: 0, n: 0 },
            expect: None,
            diverged: false,
        };
        d.init_block();
        d
    }

    #[inline]
    fn max_dist(&self) -> usize {
        self.w_size - MIN_LOOKAHEAD
    }

    #[inline]
    fn update_hash(&mut self, c: u8) {
        self.ins_h = ((self.ins_h << self.hash_shift) ^ c as usize) & self.hash_mask;
    }

    /// INSERT_STRING: returns the previous head of the hash chain.
    #[inline]
    fn insert_string(&mut self, s: usize) -> usize {
        self.update_hash(self.window[s + MIN_MATCH - 1]);
        let h = self.head[self.ins_h] as usize;
        self.prev[s & self.w_mask] = h as u16;
        self.head[self.ins_h] = s as u16;
        h
    }

    fn slide_hash(&mut self) {
        let w = self.w_size;
        for h in self.head.iter_mut() {
            *h = if *h as usize >= w { (*h as usize - w) as u16 } else { 0 };
        }
        for h in self.prev.iter_mut() {
            *h = if *h as usize >= w { (*h as usize - w) as u16 } else { 0 };
        }
    }

    fn fill_window(&mut self) {
        let wsize = self.w_size;
        loop {
            let mut more = self.window_size - self.lookahead - self.strstart;
            if self.strstart >= wsize + self.max_dist() {
                self.window.copy_within(wsize..2 * wsize - more, 0);
                self.match_start = self.match_start.wrapping_sub(wsize);
                self.strstart -= wsize;
                self.block_start -= wsize as isize;
                if self.insert > self.strstart {
                    self.insert = self.strstart;
                }
                self.slide_hash();
                more += wsize;
            }
            if self.in_pos >= self.input.len() {
                break;
            }
            let n = more.min(self.input.len() - self.in_pos);
            let at = self.strstart + self.lookahead;
            self.window[at..at + n].copy_from_slice(&self.input[self.in_pos..self.in_pos + n]);
            self.in_pos += n;
            self.lookahead += n;
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut str = self.strstart - self.insert;
                self.ins_h = self.window[str] as usize;
                self.update_hash(self.window[str + 1]);
                while self.insert > 0 {
                    self.update_hash(self.window[str + MIN_MATCH - 1]);
                    self.prev[str & self.w_mask] = self.head[self.ins_h];
                    self.head[self.ins_h] = str as u16;
                    str += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH {
                        break;
                    }
                }
            }
            if !(self.lookahead < MIN_LOOKAHEAD && self.in_pos < self.input.len()) {
                break;
            }
        }
    }

    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = self.max_chain;
        let scan = self.strstart;
        let mut best_len = self.prev_length;
        let mut nice_match = self.nice_match;
        let limit = if self.strstart > self.max_dist() { self.strstart - self.max_dist() } else { 0 };
        let w = &self.window;
        let mut scan_end1 = w[scan + best_len - 1];
        let mut scan_end = w[scan + best_len];
        if self.prev_length >= self.good_match {
            chain_length >>= 2;
        }
        if nice_match > self.lookahead {
            nice_match = self.lookahead;
        }
        loop {
            let m = cur_match;
            if !(w[m + best_len] != scan_end || w[m + best_len - 1] != scan_end1 || w[m] != w[scan] || w[m + 1] != w[scan + 1])
            {
                // bytes 0 and 1 matched; count on from 2, up to MAX_MATCH
                let mut len = 2;
                while len < MAX_MATCH && w[scan + len] == w[m + len] {
                    len += 1;
                }
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match {
                        break;
                    }
                    scan_end1 = w[scan + best_len - 1];
                    scan_end = w[scan + best_len];
                }
            }
            cur_match = self.prev[cur_match & self.w_mask] as usize;
            if cur_match <= limit {
                break;
            }
            chain_length -= 1;
            if chain_length == 0 {
                break;
            }
        }
        best_len.min(self.lookahead)
    }

    fn init_block(&mut self) {
        for n in 0..L_CODES {
            self.ltree.freq[n] = 0;
        }
        for n in 0..D_CODES {
            self.dtree.freq[n] = 0;
        }
        for n in 0..BL_CODES {
            self.bltree.freq[n] = 0;
        }
        self.ltree.freq[END_BLOCK] = 1;
        self.opt_len = 0;
        self.static_len = 0;
        self.syms.clear();
    }

    #[inline]
    fn tally_lit(&mut self, c: u8) -> bool {
        self.syms.push((0, c));
        self.ltree.freq[c as usize] += 1;
        self.syms.len() == self.sym_limit
    }

    #[inline]
    fn tally_dist(&mut self, dist: usize, len: usize) -> bool {
        let t = tables();
        self.syms.push((dist as u16, len as u8));
        let dist = dist - 1;
        self.ltree.freq[t.length_code[len] as usize + 257] += 1;
        self.dtree.freq[d_code(t, dist)] += 1;
        self.syms.len() == self.sym_limit
    }

    fn flush_block(&mut self, last: bool) {
        let stored_len = (self.strstart as isize - self.block_start) as usize;
        let buf_ok = self.block_start >= 0;
        let before = self.bw.out.len();
        self.tr_flush_block(buf_ok, stored_len, last);
        self.block_start = self.strstart as isize;
        if let Some(e) = self.expect {
            let out = &self.bw.out;
            if out.len() > e.len() || out[before..] != e[before..out.len()] {
                self.diverged = true;
            }
        }
    }

    fn deflate_fast(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            if hash_head != 0 && self.strstart - hash_head <= self.max_dist() {
                self.match_length = self.longest_match(hash_head);
            }
            let bflush;
            if self.match_length >= MIN_MATCH {
                bflush = self.tally_dist(self.strstart - self.match_start, self.match_length - MIN_MATCH);
                self.lookahead -= self.match_length;
                if self.match_length <= self.max_lazy && self.lookahead >= MIN_MATCH {
                    self.match_length -= 1;
                    loop {
                        self.strstart += 1;
                        self.insert_string(self.strstart);
                        self.match_length -= 1;
                        if self.match_length == 0 {
                            break;
                        }
                    }
                    self.strstart += 1;
                } else {
                    self.strstart += self.match_length;
                    self.match_length = 0;
                    self.ins_h = self.window[self.strstart] as usize;
                    self.update_hash(self.window[self.strstart + 1]);
                }
            } else {
                bflush = self.tally_lit(self.window[self.strstart]);
                self.lookahead -= 1;
                self.strstart += 1;
            }
            if bflush {
                self.flush_block(false);
                if self.diverged {
                    return;
                }
            }
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(true);
    }

    fn deflate_slow(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0 && self.prev_length < self.max_lazy && self.strstart - hash_head <= self.max_dist() {
                self.match_length = self.longest_match(hash_head);
                if self.match_length <= 5
                    && (self.p.strategy == Strategy::Filtered
                        || (self.match_length == MIN_MATCH && self.strstart - self.match_start > TOO_FAR))
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let bflush = self.tally_dist(self.strstart - 1 - self.prev_match, self.prev_length - MIN_MATCH);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(false);
                    if self.diverged {
                        return;
                    }
                }
            } else if self.match_available {
                let bflush = self.tally_lit(self.window[self.strstart - 1]);
                if bflush {
                    self.flush_block(false);
                    if self.diverged {
                        return;
                    }
                }
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            self.tally_lit(self.window[self.strstart - 1]);
            self.match_available = false;
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(true);
    }

    fn deflate_rle(&mut self) {
        loop {
            if self.lookahead <= MAX_MATCH {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            self.match_length = 0;
            if self.lookahead >= MIN_MATCH && self.strstart > 0 {
                let w = &self.window;
                let s = self.strstart;
                let prev = w[s - 1];
                if prev == w[s] && prev == w[s + 1] && prev == w[s + 2] {
                    let mut len = 3;
                    while len < MAX_MATCH && w[s + len] == prev {
                        len += 1;
                    }
                    self.match_length = len.min(self.lookahead);
                }
            }
            let bflush;
            if self.match_length >= MIN_MATCH {
                bflush = self.tally_dist(1, self.match_length - MIN_MATCH);
                self.lookahead -= self.match_length;
                self.strstart += self.match_length;
                self.match_length = 0;
            } else {
                bflush = self.tally_lit(self.window[self.strstart]);
                self.lookahead -= 1;
                self.strstart += 1;
            }
            if bflush {
                self.flush_block(false);
                if self.diverged {
                    return;
                }
            }
        }
        self.insert = 0;
        self.flush_block(true);
    }

    fn deflate_huff(&mut self) {
        loop {
            if self.lookahead == 0 {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            self.match_length = 0;
            let bflush = self.tally_lit(self.window[self.strstart]);
            self.lookahead -= 1;
            self.strstart += 1;
            if bflush {
                self.flush_block(false);
                if self.diverged {
                    return;
                }
            }
        }
        self.insert = 0;
        self.flush_block(true);
    }

    // --- trees.c ---

    #[inline]
    fn smaller(tree: &Tree, depth: &[u8; HEAP_SIZE], n: usize, m: usize) -> bool {
        tree.freq[n] < tree.freq[m] || (tree.freq[n] == tree.freq[m] && depth[n] <= depth[m])
    }

    fn pqdownheap(heap: &mut [usize; HEAP_SIZE], heap_len: usize, depth: &[u8; HEAP_SIZE], tree: &Tree, mut k: usize) {
        let v = heap[k];
        let mut j = k << 1;
        while j <= heap_len {
            if j < heap_len && Self::smaller(tree, depth, heap[j + 1], heap[j]) {
                j += 1;
            }
            if Self::smaller(tree, depth, v, heap[j]) {
                break;
            }
            heap[k] = heap[j];
            k = j;
            j <<= 1;
        }
        heap[k] = v;
    }

    /// zlib's build_tree + gen_bitlen + gen_codes, for tree `which` (0 lit,
    /// 1 dist, 2 bit lengths).
    fn build_tree(&mut self, which: usize) {
        let t = tables();
        let (elems, max_length, base, extra): (usize, usize, usize, &[u8]) = match which {
            0 => (L_CODES, MAX_BITS, 257, &EXTRA_LBITS),
            1 => (D_CODES, MAX_BITS, 0, &EXTRA_DBITS),
            _ => (BL_CODES, MAX_BL_BITS, 0, &EXTRA_BLBITS),
        };
        let mut tree = std::mem::replace(
            match which {
                0 => &mut self.ltree,
                1 => &mut self.dtree,
                _ => &mut self.bltree,
            },
            Tree::new(0),
        );
        let stree_len = |n: usize| -> u64 {
            match which {
                0 => t.static_ltree[n].1 as u64,
                1 => t.static_dtree[n].1 as u64,
                _ => 0,
            }
        };
        let has_stree = which < 2;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        let mut max_code: isize = -1;
        for n in 0..elems {
            if tree.freq[n] != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n;
                max_code = n as isize;
                self.depth[n] = 0;
            } else {
                tree.len[n] = 0;
            }
        }
        // the code needs at least two codes of non-zero frequency
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node;
            tree.freq[node] = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if has_stree {
                self.static_len = self.static_len.wrapping_sub(stree_len(node));
            }
        }
        tree.max_code = max_code as usize;
        let mut n = self.heap_len / 2;
        while n >= 1 {
            Self::pqdownheap(&mut self.heap, self.heap_len, &self.depth, &tree, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            // pqremove
            let n = self.heap[1];
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            Self::pqdownheap(&mut self.heap, self.heap_len, &self.depth, &tree, 1);
            let m = self.heap[1];
            self.heap_max -= 1;
            self.heap[self.heap_max] = n;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m;
            tree.freq[node] = tree.freq[n].wrapping_add(tree.freq[m]);
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            tree.dad[n] = node as u16;
            tree.dad[m] = node as u16;
            self.heap[1] = node;
            node += 1;
            Self::pqdownheap(&mut self.heap, self.heap_len, &self.depth, &tree, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];

        // gen_bitlen
        for b in self.bl_count.iter_mut() {
            *b = 0;
        }
        tree.len[self.heap[self.heap_max]] = 0;
        let mut overflow = 0i32;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h];
            let mut bits = tree.len[tree.dad[n] as usize] as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree.len[n] = bits as u8;
            h += 1;
            if n > tree.max_code {
                continue; // not a leaf
            }
            self.bl_count[bits] += 1;
            let xbits = if n >= base { extra[n - base] as u64 } else { 0 };
            let f = tree.freq[n] as u64;
            self.opt_len = self.opt_len.wrapping_add(f * (bits as u64 + xbits));
            if has_stree {
                self.static_len = self.static_len.wrapping_add(f * (stree_len(n) + xbits));
            }
        }
        if overflow > 0 {
            loop {
                let mut bits = max_length - 1;
                while self.bl_count[bits] == 0 {
                    bits -= 1;
                }
                self.bl_count[bits] -= 1;
                self.bl_count[bits + 1] += 2;
                self.bl_count[max_length] -= 1;
                overflow -= 2;
                if overflow <= 0 {
                    break;
                }
            }
            let mut h = HEAP_SIZE;
            let mut bits = max_length;
            while bits != 0 {
                let mut n = self.bl_count[bits];
                while n != 0 {
                    h -= 1;
                    let m = self.heap[h];
                    if m > tree.max_code {
                        continue;
                    }
                    if tree.len[m] as usize != bits {
                        self.opt_len = self
                            .opt_len
                            .wrapping_add((bits as u64).wrapping_sub(tree.len[m] as u64).wrapping_mul(tree.freq[m] as u64));
                        tree.len[m] = bits as u8;
                    }
                    n -= 1;
                }
                bits -= 1;
            }
        }
        let mc = tree.max_code;
        let (lens, codes) = (&tree.len, &mut tree.code);
        gen_codes(lens, mc, &self.bl_count, codes);
        match which {
            0 => self.ltree = tree,
            1 => self.dtree = tree,
            _ => self.bltree = tree,
        }
    }

    /// Length of code `n` in `tree`, with zlib's 0xffff guard one past max_code.
    #[inline]
    fn tlen(tree: &Tree, n: usize, max_code: usize) -> u32 {
        if n == max_code + 1 {
            0xffff
        } else {
            tree.len[n] as u32
        }
    }

    fn scan_tree(&mut self, which: usize) {
        let tree = if which == 0 { &self.ltree } else { &self.dtree };
        let max_code = tree.max_code;
        let mut prevlen: i32 = -1;
        let mut nextlen = tree.len[0] as u32;
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        let mut add = [0u16; BL_CODES];
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = Self::tlen(tree, n + 1, max_code);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                add[curlen as usize] += count as u16;
            } else if curlen != 0 {
                if curlen as i32 != prevlen {
                    add[curlen as usize] += 1;
                }
                add[REP_3_6] += 1;
            } else if count <= 10 {
                add[REPZ_3_10] += 1;
            } else {
                add[REPZ_11_138] += 1;
            }
            count = 0;
            prevlen = curlen as i32;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
        for i in 0..BL_CODES {
            self.bltree.freq[i] += add[i];
        }
    }

    fn send_code(bw: &mut BitWriter, tree: &Tree, c: usize) {
        bw.send(tree.code[c] as u32, tree.len[c] as u32);
    }

    fn send_tree(&mut self, which: usize) {
        let tree = if which == 0 { &self.ltree } else { &self.dtree };
        let bl = &self.bltree;
        let bw = &mut self.bw;
        let max_code = tree.max_code;
        let mut prevlen: i32 = -1;
        let mut nextlen = tree.len[0] as u32;
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = Self::tlen(tree, n + 1, max_code);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                while count > 0 {
                    Self::send_code(bw, bl, curlen as usize);
                    count -= 1;
                }
            } else if curlen != 0 {
                if curlen as i32 != prevlen {
                    Self::send_code(bw, bl, curlen as usize);
                    count -= 1;
                }
                Self::send_code(bw, bl, REP_3_6);
                bw.send(count - 3, 2);
            } else if count <= 10 {
                Self::send_code(bw, bl, REPZ_3_10);
                bw.send(count - 3, 3);
            } else {
                Self::send_code(bw, bl, REPZ_11_138);
                bw.send(count - 11, 7);
            }
            count = 0;
            prevlen = curlen as i32;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    fn build_bl_tree(&mut self) -> usize {
        self.scan_tree(0);
        self.scan_tree(1);
        self.build_tree(2);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bltree.len[BL_ORDER[max_blindex]] != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.opt_len = self.opt_len.wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        max_blindex
    }

    fn compress_block(&mut self, fixed: bool) {
        let t = tables();
        let (lt, dt) = (&self.ltree, &self.dtree);
        let bw = &mut self.bw;
        let lcode = |c: usize| if fixed { t.static_ltree[c] } else { (lt.code[c], lt.len[c]) };
        let dcode = |c: usize| if fixed { t.static_dtree[c] } else { (dt.code[c], dt.len[c]) };
        for &(dist, lc) in &self.syms {
            if dist == 0 {
                let (c, l) = lcode(lc as usize);
                bw.send(c as u32, l as u32);
            } else {
                let code = t.length_code[lc as usize] as usize;
                let (c, l) = lcode(code + 257);
                bw.send(c as u32, l as u32);
                let extra = EXTRA_LBITS[code] as u32;
                if extra != 0 {
                    bw.send(lc as u32 - t.base_length[code] as u32, extra);
                }
                let dist = dist as usize - 1;
                let code = d_code(t, dist);
                let (c, l) = dcode(code);
                bw.send(c as u32, l as u32);
                let extra = EXTRA_DBITS[code] as u32;
                if extra != 0 {
                    bw.send((dist - t.base_dist[code] as usize) as u32, extra);
                }
            }
        }
        let (c, l) = lcode(END_BLOCK);
        bw.send(c as u32, l as u32);
    }

    fn tr_flush_block(&mut self, buf_ok: bool, stored_len: usize, last: bool) {
        let t = tables();
        self.build_tree(0);
        self.build_tree(1);
        let max_blindex = self.build_bl_tree();
        let mut opt_lenb = (self.opt_len.wrapping_add(3 + 7)) >> 3;
        let static_lenb = (self.static_len.wrapping_add(3 + 7)) >> 3;
        if static_lenb <= opt_lenb || self.p.strategy == Strategy::Fixed {
            opt_lenb = static_lenb;
        }
        let last_bit = last as u32;
        if stored_len as u64 + 4 <= opt_lenb && buf_ok {
            // stored block
            self.bw.send(last_bit, 3);
            self.bw.windup();
            self.bw.out.extend_from_slice(&(stored_len as u16).to_le_bytes());
            self.bw.out.extend_from_slice(&(!(stored_len as u16)).to_le_bytes());
            let start = self.block_start as usize;
            self.bw.out.extend_from_slice(&self.window[start..start + stored_len]);
        } else if static_lenb == opt_lenb {
            self.bw.send((1 << 1) + last_bit, 3);
            self.compress_block(true);
        } else {
            self.bw.send((2 << 1) + last_bit, 3);
            let lcodes = self.ltree.max_code + 1;
            let dcodes = self.dtree.max_code + 1;
            let blcodes = max_blindex + 1;
            self.bw.send(lcodes as u32 - 257, 5);
            self.bw.send(dcodes as u32 - 1, 5);
            self.bw.send(blcodes as u32 - 4, 4);
            for &o in BL_ORDER.iter().take(blcodes) {
                let l = self.bltree.len[o] as u32;
                self.bw.send(l, 3);
            }
            self.send_tree(0);
            self.send_tree(1);
            self.compress_block(false);
        }
        let _ = t;
        self.init_block();
        if last {
            self.bw.windup();
        }
    }
}

fn run(d: &mut Deflater) {
    match (d.p.strategy, d.p.level) {
        (Strategy::HuffmanOnly, _) => d.deflate_huff(),
        (Strategy::Rle, _) => d.deflate_rle(),
        (_, 1..=3) => d.deflate_fast(),
        _ => d.deflate_slow(),
    }
}

/// Compress `data` exactly as zlib would with `p`, as a raw deflate stream
/// (no zlib or gzip wrapper).
pub fn deflate(data: &[u8], p: Params) -> Vec<u8> {
    let mut d = Deflater::new(data, p);
    run(&mut d);
    d.bw.out
}

/// Does deflating `data` with `p` give exactly `expected`? Gives up at the
/// first block that differs, so a wrong guess costs one block, not a stream.
pub fn reproduces(data: &[u8], p: Params, expected: &[u8]) -> bool {
    let mut d = Deflater::new(data, p);
    d.expect = Some(expected);
    run(&mut d);
    !d.diverged && d.bw.out == expected
}

/// Find zlib parameters that reproduce `expected` from `data`, trying the
/// common settings first. `wbits` is the window the stream declares (15 when
/// a container doesn't say).
pub fn find_params(data: &[u8], expected: &[u8], wbits: u8) -> Option<Params> {
    let mut tried: Vec<Params> = Vec::new();
    let mut try_p = |p: Params| -> bool {
        if tried.contains(&p) {
            return false;
        }
        tried.push(p);
        reproduces(data, p, expected)
    };
    let levels = [6u8, 9, 1, 5, 4, 3, 2, 7, 8];
    for &mem_level in &[8u8, 9, 7, 6, 5, 4, 3, 2, 1] {
        for &strategy in &[Strategy::Default, Strategy::Filtered] {
            for &level in &levels {
                let p = Params { level, mem_level, wbits, strategy };
                if try_p(p) {
                    return Some(p);
                }
            }
        }
        for &strategy in &[Strategy::Rle, Strategy::HuffmanOnly, Strategy::Fixed] {
            for &level in if strategy == Strategy::Fixed { &levels[..] } else { &levels[..1] } {
                let p = Params { level, mem_level, wbits, strategy };
                if try_p(p) {
                    return Some(p);
                }
            }
        }
        // only the default memLevel is common; the rest are a long tail
        if mem_level == 9 {
            break;
        }
    }
    None
}

/// What zlib would write as the dynamic-block header for these symbol
/// frequencies (`lit` 286 entries with END_BLOCK counted, `dist` 30), and the
/// code lengths it would assign. The header is returned as (bytes, bit count),
/// LSB-first, starting right after the 3-bit block type.
pub fn zlib_dynamic_header(lit: &[u16], dist: &[u16]) -> (Vec<u8>, usize, Vec<u8>, Vec<u8>) {
    let p = Params { level: 6, mem_level: 8, wbits: 15, strategy: Strategy::Default };
    let mut d = Deflater::new(&[], p);
    d.ltree.freq[..L_CODES].copy_from_slice(&lit[..L_CODES]);
    d.dtree.freq[..D_CODES].copy_from_slice(&dist[..D_CODES]);
    d.build_tree(0);
    d.build_tree(1);
    let max_blindex = d.build_bl_tree();
    let lcodes = d.ltree.max_code + 1;
    let dcodes = d.dtree.max_code + 1;
    let blcodes = max_blindex + 1;
    d.bw.send(lcodes as u32 - 257, 5);
    d.bw.send(dcodes as u32 - 1, 5);
    d.bw.send(blcodes as u32 - 4, 4);
    for &o in BL_ORDER.iter().take(blcodes) {
        let l = d.bltree.len[o] as u32;
        d.bw.send(l, 3);
    }
    d.send_tree(0);
    d.send_tree(1);
    let nbits = d.bw.out.len() * 8 + d.bw.n as usize;
    let llens = d.ltree.len[..L_CODES].to_vec();
    let dlens = d.dtree.len[..D_CODES].to_vec();
    d.bw.windup();
    (d.bw.out, nbits, llens, dlens)
}

