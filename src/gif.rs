//! GIF: undo the LZW so the pixels can be modelled, then redo it exactly.
//!
//! GIF's LZW leaves an encoder almost no freedom: the parse is always greedy
//! (emit the longest string in the dictionary, add it plus the next pixel), and
//! the code width follows the dictionary size. What an encoder does choose is
//! where to emit clear codes (dictionary resets — when the table fills, or
//! earlier), whether to open with one, whether to close with an end code, and
//! how to cut the output into sub-blocks. Decoding records exactly those
//! choices; re-encoding replays them, and the result is checked against the
//! original bits before anything is trusted.

/// What an image's LZW stream needs beyond its pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Lzw {
    pub min_size: u8,
    pub clears: Vec<u32>,  // pixel counts at which a clear code was emitted
    /// Where the encoder emitted a shorter string than the longest it had:
    /// (pixel position the string starts at, its length).
    pub cuts: Vec<(u32, u16)>,
    pub end_code: bool,    // closes with an end-of-information code
    pub blocks: Vec<u8>,   // sub-block sizes, all but the last (usually 255s)
    pub last_block: u8,
}

/// Width of code `t` (counting from the last clear): the decoder adds one
/// entry per code after the first, and widens when the next entry no longer
/// fits — so code t is read at the smallest width holding clear + 1 + t.
#[inline]
fn width_for(t: u32, clear: u32, min_size: u8) -> u32 {
    let mut w = min_size as u32 + 1;
    if t > 0 {
        let next = (clear + 1 + t).min(4095);
        while w < 12 && (1 << w) <= next {
            w += 1;
        }
    }
    w
}

/// Decode an LZW stream: (pixels, clear positions, ended with an end code).
fn read_codes(data: &[u8], min_size: u8, limit: usize) -> Option<(Vec<u8>, Vec<u32>, bool, Vec<u16>)> {
    let clear = 1u32 << min_size;
    let eoi = clear + 1;
    let mut prefix = vec![0u16; 4096];
    let mut suffix = vec![0u8; 4096];
    let mut first = vec![0u8; 4096];
    for i in 0..clear {
        suffix[i as usize] = i as u8;
        first[i as usize] = i as u8;
    }
    let mut out: Vec<u8> = Vec::new();
    let mut clears = Vec::new();
    let mut lens: Vec<u16> = Vec::new();
    let mut prev: Option<u32> = None;
    let mut next = clear + 2;
    let mut t = 0u32; // codes since the last clear
    let (mut acc, mut nbits, mut pos) = (0u64, 0u32, 0usize);
    let mut stack: Vec<u8> = Vec::with_capacity(4096);
    loop {
        let width = width_for(t, clear, min_size);
        while nbits < width {
            if pos >= data.len() {
                // ran out without an end code: fine if only zero padding is left
                return (acc == 0).then_some((out, clears, false, lens));
            }
            acc |= (data[pos] as u64) << nbits;
            pos += 1;
            nbits += 8;
        }
        let code = (acc & ((1 << width) - 1)) as u32;
        acc >>= width;
        nbits -= width;
        if code == clear {
            clears.push(out.len() as u32);
            next = clear + 2;
            prev = None;
            t = 0;
            continue;
        }
        if code == eoi {
            // what follows must be padding the encoder will recreate
            let rest_ok = acc == 0 && pos == data.len();
            return rest_ok.then_some((out, clears, true, lens));
        }
        let kwkwk = code == next;
        if code > next || (kwkwk && prev.is_none()) || (code >= clear && code < clear + 2) || (code >= next && next >= 4096) {
            return None;
        }
        let base = if kwkwk { prev.unwrap() } else { code };
        stack.clear();
        let mut c = base;
        while c >= clear + 2 {
            stack.push(suffix[c as usize]);
            c = prefix[c as usize] as u32;
        }
        stack.push(c as u8);
        let lead = c as u8; // first byte of string(base)
        out.extend(stack.iter().rev());
        if kwkwk {
            out.push(lead);
        }
        lens.push((stack.len() + kwkwk as usize) as u16);
        if out.len() > limit {
            return None;
        }
        if let Some(p) = prev {
            if next < 4096 {
                prefix[next as usize] = p as u16;
                suffix[next as usize] = lead;
                first[next as usize] = first[p as usize];
                next += 1;
            }
        }
        prev = Some(code);
        t += 1;
    }
}

/// The LZW parse, one string at a time: greedy, except where a recorded cut
/// or clear ends a string early. `on_string` sees each string's start and
/// greedy length before the cut is applied; it returns the length to use.
/// Returns the number of codes since the last clear, which sets the end
/// code's width.
fn parse(pixels: &[u8], l: &Lzw, mut on_string: impl FnMut(usize, usize) -> Option<usize>, mut emit: impl FnMut(u32, u32)) -> Option<u32> {
    let clear = 1u32 << l.min_size;
    let mut dict: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut next = clear + 2;
    let mut t = 0u32;
    let (mut i, mut ci) = (0usize, 0usize);
    let n = pixels.len();
    loop {
        while ci < l.clears.len() && l.clears[ci] as usize == i {
            emit(clear, width_for(t, clear, l.min_size));
            dict.clear();
            next = clear + 2;
            t = 0;
            ci += 1;
        }
        if i == n {
            break;
        }
        // a string never runs past the next clear
        let stop = if ci < l.clears.len() { (l.clears[ci] as usize).min(n) } else { n };
        let (mut code, mut len) = (pixels[i] as u32, 1usize);
        let mut chain: Vec<u32> = Vec::new();
        chain.push(code);
        while i + len < stop {
            match dict.get(&(code << 8 | pixels[i + len] as u32)) {
                Some(&e) => {
                    code = e;
                    len += 1;
                    chain.push(code);
                }
                None => break,
            }
        }
        let take = on_string(i, len)?;
        if take == 0 || take > len {
            return None;
        }
        code = chain[take - 1];
        emit(code, width_for(t, clear, l.min_size));
        t += 1;
        if i + take < stop && next < 4096 {
            dict.insert(code << 8 | pixels[i + take] as u32, next);
            next += 1;
        }
        i += take;
    }
    Some(t)
}

struct Packer {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl Packer {
    fn put(&mut self, code: u32, width: u32) {
        self.acc |= (code as u64) << self.nbits;
        self.nbits += width;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
}

/// LZW-code `pixels` with the recorded choices; returns the packed code stream.
pub fn encode(pixels: &[u8], l: &Lzw) -> Vec<u8> {
    let clear = 1u32 << l.min_size;
    let mut pk = Packer { out: Vec::with_capacity(pixels.len() / 2), acc: 0, nbits: 0 };
    let mut ki = 0usize;
    let cuts = &l.cuts;
    let on = |start: usize, greedy: usize| -> Option<usize> {
        if ki < cuts.len() && cuts[ki].0 as usize == start {
            ki += 1;
            Some(cuts[ki - 1].1 as usize)
        } else {
            Some(greedy)
        }
    };
    let Some(t) = parse(pixels, l, on, |c, w| pk.put(c, w)) else { return Vec::new() };
    if l.end_code {
        pk.put(clear + 1, width_for(t, clear, l.min_size));
    }
    if pk.nbits > 0 {
        pk.out.push(pk.acc as u8);
    }
    pk.out
}

/// Where an encoder's strings fall short of greedy, given the lengths it
/// actually used (from decoding). None if it ever ran longer than greedy,
/// which no LZW decoder could have read.
fn derive_cuts(pixels: &[u8], l: &Lzw, lens: &[u16]) -> Option<Vec<(u32, u16)>> {
    let mut cuts = Vec::new();
    let mut k = 0usize;
    let on = |start: usize, greedy: usize| -> Option<usize> {
        let want = *lens.get(k)? as usize;
        k += 1;
        if want < greedy {
            cuts.push((start as u32, want as u16));
        }
        (want <= greedy).then_some(want)
    };
    parse(pixels, l, on, |_, _| {})?;
    Some(cuts)
}

/// Cut a code stream into GIF sub-blocks the way the original was cut.
pub fn blocks(codes: &[u8], l: &Lzw) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(codes.len() + codes.len() / 255 + 2);
    let mut at = 0;
    for &n in &l.blocks {
        let n = n as usize;
        out.push(n as u8);
        out.extend_from_slice(codes.get(at..at + n)?);
        at += n;
    }
    if codes.len() - at != l.last_block as usize {
        return None;
    }
    if l.last_block > 0 {
        out.push(l.last_block);
        out.extend_from_slice(&codes[at..]);
    }
    out.push(0);
    Some(out)
}

/// One image found in a GIF.
pub struct Image {
    pub data_start: usize, // offset of the sub-blocks (after the min-size byte)
    pub data_end: usize,   // just past the terminating zero-length block
    pub lzw: Lzw,
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub interlaced: bool,
}

/// Walk a GIF; returns its length and every image whose LZW we can redo.
pub fn scan(d: &[u8]) -> Option<(usize, Vec<Image>)> {
    if d.len() < 13 || !(d.starts_with(b"GIF87a") || d.starts_with(b"GIF89a")) {
        return None;
    }
    let mut p = 13;
    if d[10] & 0x80 != 0 {
        p += 3 << ((d[10] & 7) + 1);
    }
    let mut images = Vec::new();
    loop {
        match *d.get(p)? {
            0x3b => return Some((p + 1, images)),
            0x21 => {
                p += 2;
                loop {
                    let n = *d.get(p)? as usize;
                    p += 1 + n;
                    if n == 0 {
                        break;
                    }
                }
            }
            0x2c => {
                let desc = d.get(p..p + 10)?;
                let w = u16::from_le_bytes([desc[5], desc[6]]) as usize;
                let h = u16::from_le_bytes([desc[7], desc[8]]) as usize;
                let flags = desc[9];
                p += 10;
                if flags & 0x80 != 0 {
                    p += 3 << ((flags & 7) + 1);
                }
                let min_size = *d.get(p)?;
                p += 1;
                let data_start = p;
                let mut codes = Vec::new();
                let mut sizes = Vec::new();
                loop {
                    let n = *d.get(p)? as usize;
                    if n == 0 {
                        p += 1;
                        break;
                    }
                    codes.extend_from_slice(d.get(p + 1..p + 1 + n)?);
                    sizes.push(n as u8);
                    p += 1 + n;
                }
                if !(2..=8).contains(&min_size) {
                    continue;
                }
                let Some((pixels, clears, end_code, lens)) = read_codes(&codes, min_size, w * h + 1) else { continue };
                let last_block = sizes.pop().unwrap_or(0);
                let mut lzw = Lzw { min_size, clears, cuts: Vec::new(), end_code, blocks: sizes, last_block };
                let Some(cuts) = derive_cuts(&pixels, &lzw, &lens) else { continue };
                lzw.cuts = cuts;
                // only if it comes back bit for bit
                let ok = pixels.len() == w * h && blocks(&encode(&pixels, &lzw), &lzw).as_deref() == Some(&d[data_start..p]);
                if ok {
                    images.push(Image { data_start, data_end: p, lzw, pixels, width: w, height: h, interlaced: flags & 0x40 != 0 });
                }
            }
            _ => return None,
        }
    }
}

/// Interlaced row order: rows 0,8,16.. then 4,12.. then 2,6.. then 1,3..
pub fn interlace_order(h: usize) -> Vec<usize> {
    let mut v = Vec::with_capacity(h);
    for &(start, step) in &[(0, 8), (4, 8), (2, 4), (1, 2)] {
        let mut r = start;
        while r < h {
            v.push(r);
            r += step;
        }
    }
    v
}

/// Analysis aid: for each image, why it does or doesn't round-trip.
pub fn diagnose(d: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut p = 13;
    if d.len() < 13 {
        return out;
    }
    if d[10] & 0x80 != 0 {
        p += 3 << ((d[10] & 7) + 1);
    }
    while p < d.len() {
        match d[p] {
            0x21 => {
                p += 2;
                while p < d.len() {
                    let n = d[p] as usize;
                    p += 1 + n;
                    if n == 0 {
                        break;
                    }
                }
            }
            0x2c => {
                let w = u16::from_le_bytes([d[p + 5], d[p + 6]]) as usize;
                let h = u16::from_le_bytes([d[p + 7], d[p + 8]]) as usize;
                let flags = d[p + 9];
                p += 10;
                if flags & 0x80 != 0 {
                    p += 3 << ((flags & 7) + 1);
                }
                let min_size = d[p];
                p += 1;
                let st = p;
                let mut codes = Vec::new();
                let mut sizes = Vec::new();
                loop {
                    let n = d[p] as usize;
                    if n == 0 {
                        p += 1;
                        break;
                    }
                    codes.extend_from_slice(&d[p + 1..p + 1 + n]);
                    sizes.push(n as u8);
                    p += 1 + n;
                }
                match read_codes(&codes, min_size, w * h + 1) {
                    None => out.push(format!("{w}x{h} min {min_size}: decode failed")),
                    Some((px, clears, end, lens)) => {
                        let last_block = sizes.pop().unwrap_or(0);
                        let mut l = Lzw { min_size, clears: clears.clone(), cuts: Vec::new(), end_code: end, blocks: sizes, last_block };
                        l.cuts = derive_cuts(&px, &l, &lens).unwrap_or_default();
                        out.push(format!("cuts: {}", l.cuts.len()));
                        let re = encode(&px, &l);
                        let diff = re.iter().zip(&codes).position(|(a, b)| a != b);
                        out.push(format!(
                            "{w}x{h} min {min_size}: {} px, clears {:?}.., end {end}, re-encode {} vs {} bytes, first diff {:?}, blocks {:?}",
                            px.len(), &clears[..clears.len().min(4)], re.len(), codes.len(), diff,
                            blocks(&re, &l).map(|b| b == d[st..p])
                        ));
                    }
                }
            }
            _ => break,
        }
    }
    out
}

