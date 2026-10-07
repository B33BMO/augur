//! Recompression: expand a file into what its compressed streams hide.
//!
//! `expand` walks a file looking for deflate streams — PNG image data, zlib
//! streams (PDF, git, embedded resources), gzip members, ZIP entries — and for
//! each one that `deflate::find_params` can reproduce exactly, replaces the
//! compressed bytes with the decompressed ones. The result is a *virtual*
//! stream, which the CM codes, and a *recipe*: a tree of pieces that `rebuild`
//! applies to turn the virtual stream back into the original file, bit for bit.
//!
//! Expansion recurses (a gzip of a tar of PNGs unpacks all the way down), and
//! PNG scanlines are unfiltered back into pixels, which become a sample layout
//! for the image model — so a PNG is coded as an image, not as deflate output.

use super::deflate::{self, Params, Strategy};
use super::front::{self, Layout};

/// A transform applied to a span of the virtual stream on rebuild.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// The span (once its children are rebuilt) is deflated with these params.
    Deflate(Params),
    /// The span is a zlib stream to be cut into PNG IDAT chunks of these
    /// payload sizes, each with its length, type and CRC.
    Idat(Vec<u32>),
    /// The span is a reflate diff of `diff_len` bytes followed by the data;
    /// rebuild re-encodes the deflate stream the diff describes.
    Reflate { diff_len: usize },
    /// The span is a GIF image's pixels (in display order); rebuild LZW-codes
    /// them with the recorded choices and cuts the sub-blocks.
    Gif { lzw: super::gif::Lzw, width: usize, height: usize, interlaced: bool },
    /// The span is `rows` filter-type bytes followed by `rows` rows of
    /// `rowbytes` unfiltered bytes; rebuild re-applies the PNG filters.
    Unfilter { rowbytes: usize, rows: usize, bpp: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub start: usize,    // where its span begins in the virtual stream
    pub virt_len: usize, // span of the virtual stream it covers
    pub kind: Kind,
    pub children: Vec<Piece>,
}

pub struct Expanded {
    pub virt: Vec<u8>,
    pub pieces: Vec<Piece>,
    pub layouts: Vec<Layout>,
}

/// Deepest nesting followed (zip -> png -> ...); deeper streams stay packed.
const MAX_DEPTH: u32 = 4;
/// Streams shorter than this aren't worth a recipe entry.
const MIN_STREAM: usize = 64;
/// Refuse to inflate beyond this; a zip bomb must not take the machine down.
const INFLATE_LIMIT: usize = 1 << 31;

pub fn expand(data: &[u8]) -> Expanded {
    // a raw sample format at the top level is coded in place
    if let Some(l) = super::wav_parse(data).or_else(|| super::image_parse(data)) {
        return Expanded { virt: data.to_vec(), pieces: Vec::new(), layouts: vec![l] };
    }
    let mut virt = Vec::with_capacity(data.len());
    let mut layouts = Vec::new();
    let pieces = expand_into(data, 0, &mut virt, &mut layouts);
    Expanded { virt, pieces, layouts }
}

/// Expand `data`, appending its virtual form to `virt`; returns its pieces.
fn expand_into(data: &[u8], depth: u32, virt: &mut Vec<u8>, lays: &mut Vec<Layout>) -> Vec<Piece> {
    if depth > 0 {
        // a whole nested file that is itself a known raw format
        if let Some(l) = super::wav_parse(data).or_else(|| super::image_parse(data)) {
            let mut l = l;
            l.off += virt.len();
            lays.push(l);
            virt.extend_from_slice(data);
            return Vec::new();
        }
    }
    if depth >= MAX_DEPTH {
        virt.extend_from_slice(data);
        return Vec::new();
    }
    if let Some(p) = png_expand(data, virt, lays) {
        return p;
    }
    let mut pieces = Vec::new();
    let mut raw_from = 0; // first byte not yet emitted
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0x89 && data[i..].starts_with(PNG_SIG) {
            // an embedded PNG (a stored zip entry, a resource section, a PDF)
            if let Some(end) = png_end(&data[i..]) {
                virt.extend_from_slice(&data[raw_from..i]);
                if let Some(mut ps) = png_expand(&data[i..i + end], virt, lays) {
                    pieces.append(&mut ps);
                } else {
                    virt.extend_from_slice(&data[i..i + end]);
                }
                raw_from = i + end;
                i += end;
                continue;
            }
        }
        if data[i] == b'G' && (data[i..].starts_with(b"GIF87a") || data[i..].starts_with(b"GIF89a")) {
            if let Some((end, images)) = super::gif::scan(&data[i..]) {
                virt.extend_from_slice(&data[raw_from..i]);
                let mut at = 0;
                for im in images {
                    virt.extend_from_slice(&data[i + at..i + im.data_start]);
                    let start = virt.len();
                    // interlaced rows go back to display order, where the
                    // row above is the row above
                    let pixels = if im.interlaced {
                        let mut v = vec![0u8; im.pixels.len()];
                        for (k, &r) in super::gif::interlace_order(im.height).iter().enumerate() {
                            v[r * im.width..(r + 1) * im.width].copy_from_slice(&im.pixels[k * im.width..(k + 1) * im.width]);
                        }
                        v
                    } else {
                        im.pixels
                    };
                    if im.width * im.height >= 1024 && im.height >= 2 {
                        lays.push(Layout {
                            kind: front::KIND_IMAGE,
                            off: start,
                            count: im.width * im.height,
                            width: 1,
                            shift: 0,
                            flags: front::LAY_PALETTE,
                            chans: 1,
                            row: im.width,
                            stride: im.width,
                        });
                    }
                    virt.extend_from_slice(&pixels);
                    let kind = Kind::Gif { lzw: im.lzw, width: im.width, height: im.height, interlaced: im.interlaced };
                    pieces.push(Piece { start, virt_len: virt.len() - start, kind, children: Vec::new() });
                    at = im.data_end;
                }
                virt.extend_from_slice(&data[i + at..i + end]);
                raw_from = i + end;
                i += end;
                continue;
            }
        }
        if data[i] == 0xff && data[i + 1] == 0xd8 && data[i + 2] == 0xff {
            // a JPEG: its scans become passthrough regions for the JPEG model
            if let Some((end, scans)) = jpeg_scans(&data[i..]) {
                virt.extend_from_slice(&data[raw_from..i]);
                let base = virt.len();
                for (off, len) in scans {
                    lays.push(Layout {
                        kind: front::KIND_JPEG,
                        off: base + off,
                        count: len,
                        width: 1,
                        shift: 0,
                        flags: 0,
                        chans: 1,
                        row: 0,
                        stride: off,
                    });
                }
                virt.extend_from_slice(&data[i..i + end]);
                raw_from = i + end;
                i += end;
                continue;
            }
        }
        let found = if data[i] & 0x0f == 8 && data[i] >> 4 <= 7 {
            zlib_at(data, i)
        } else if data[i] == 0x1f && data[i + 1] == 0x8b {
            gzip_at(data, i)
        } else if data[i] == b'P' && data.get(i + 1..i + 4) == Some(b"K\x03\x04") {
            zip_at(data, i)
        } else {
            None
        };
        let Some((start, end, method, inflated)) = found else {
            i += 1;
            continue;
        };
        // [start, end) is a raw deflate stream we know how to remake
        virt.extend_from_slice(&data[raw_from..start]);
        pieces.push(emit(method, virt, |v| expand_into(&inflated, depth + 1, v, lays)));
        raw_from = end;
        i = end;
    }
    virt.extend_from_slice(&data[raw_from..]);
    pieces
}

/// How a deflate stream will be regenerated.
enum Method {
    Zlib(Params),
    Reflate(Vec<u8>),
}

/// A deflate stream found at [start, end), how to remake it, and its contents.
type Found = (usize, usize, Method, Vec<u8>);

/// Inflate the raw deflate stream at `at` and find how to reproduce it: zlib
/// parameters if any reproduce it, a reflate diff otherwise.
fn try_stream(data: &[u8], at: usize, wbits: u8) -> Option<Found> {
    let inf = deflate::inflate(&data[at..], INFLATE_LIMIT)?;
    if inf.consumed < MIN_STREAM {
        return None;
    }
    let comp = &data[at..at + inf.consumed];
    if let Some(p) = deflate::find_params(&inf.data, comp, wbits) {
        return Some((at, at + inf.consumed, Method::Zlib(p), inf.data));
    }
    let (raw, diff, used) = super::reflate::analyze(comp, INFLATE_LIMIT)?;
    // Encoders zlib-like enough leave a tiny diff, and unpacking is a clear
    // win. A big diff (an unusual encoder) might cost more than it frees, and
    // only a trial encode can say: both versions through a fresh model.
    if diff.len() * 8 > comp.len() {
        let mut both = diff.clone();
        both.extend_from_slice(&raw);
        let mb = super::mem_bits_for(both.len());
        let unpacked = super::encode_stream(&both, super::Mode::Generic, mb).len();
        let packed = super::encode_stream(&comp[..used], super::Mode::Generic, super::mem_bits_for(used)).len();
        if unpacked >= packed {
            return None;
        }
    }
    Some((at, at + used, Method::Reflate(diff), raw))
}

/// Emit a found stream into `virt` (diff first, for reflate) and return its
/// piece; `inner` expands the contents and returns their pieces.
fn emit(method: Method, virt: &mut Vec<u8>, inner: impl FnOnce(&mut Vec<u8>) -> Vec<Piece>) -> Piece {
    let start = virt.len();
    let kind = match method {
        Method::Zlib(p) => Kind::Deflate(p),
        Method::Reflate(diff) => {
            virt.extend_from_slice(&diff);
            Kind::Reflate { diff_len: diff.len() }
        }
    };
    let children = inner(virt);
    Piece { start, virt_len: virt.len() - start, kind, children }
}

/// A zlib stream: CMF, FLG, deflate data, Adler-32. Only the deflate part
/// becomes a piece; the wrapper bytes stay in the virtual stream verbatim.
fn zlib_at(data: &[u8], i: usize) -> Option<Found> {
    let (cmf, flg) = (data[i], data[i + 1]);
    if cmf & 0x0f != 8 || cmf >> 4 > 7 || (cmf as u16 * 256 + flg as u16) % 31 != 0 || flg & 0x20 != 0 {
        return None;
    }
    let r = try_stream(data, i + 2, (cmf >> 4) + 8)?;
    // the Adler-32 trailer must follow and match, or this was a false start
    let a = data.get(r.1..r.1 + 4)?;
    (u32::from_be_bytes(a.try_into().ok()?) == adler32(&r.3)).then_some(r)
}

/// A gzip member: 10-byte header, optional fields, deflate data, CRC, size.
fn gzip_at(data: &[u8], i: usize) -> Option<Found> {
    let h = data.get(i..i + 10)?;
    if h[2] != 8 || h[3] & 0xe0 != 0 {
        return None;
    }
    let flags = h[3];
    let mut p = i + 10;
    if flags & 4 != 0 {
        let xlen = u16::from_le_bytes([*data.get(p)?, *data.get(p + 1)?]) as usize;
        p += 2 + xlen;
    }
    for bit in [8u8, 16] {
        if flags & bit != 0 {
            while *data.get(p)? != 0 {
                p += 1;
            }
            p += 1;
        }
    }
    if flags & 2 != 0 {
        p += 2;
    }
    let r = try_stream(data, p, 15)?;
    let t = data.get(r.1..r.1 + 8)?;
    (u32::from_le_bytes(t[0..4].try_into().ok()?) == crc32(&r.3)).then_some(r)
}

/// A ZIP local file entry using deflate (method 8).
fn zip_at(data: &[u8], i: usize) -> Option<Found> {
    let h = data.get(i..i + 30)?;
    if u16::from_le_bytes([h[8], h[9]]) != 8 {
        return None;
    }
    let crc = u32::from_le_bytes(h[14..18].try_into().ok()?);
    let name_len = u16::from_le_bytes([h[26], h[27]]) as usize;
    let extra_len = u16::from_le_bytes([h[28], h[29]]) as usize;
    let r = try_stream(data, i + 30 + name_len + extra_len, 15)?;
    // with a data descriptor (flag bit 3) the header CRC may be zero
    let ok = crc == crc32(&r.3) || u16::from_le_bytes([h[6], h[7]]) & 8 != 0;
    ok.then_some(r)
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

const PNG_SIG: &[u8] = b"\x89PNG\r\n\x1a\n";

/// Expand a whole PNG: IDAT chunks become one zlib stream, its deflate data
/// is inflated, and the scanlines are unfiltered into pixels.
fn png_expand(data: &[u8], virt: &mut Vec<u8>, lays: &mut Vec<Layout>) -> Option<Vec<Piece>> {
    if !data.starts_with(PNG_SIG) {
        return None;
    }
    // walk the chunks; find IHDR and the (contiguous) run of IDATs
    let mut pos = 8;
    let mut ihdr: Option<&[u8]> = None;
    let mut idat: Option<(usize, usize, Vec<u32>)> = None; // first chunk start, end, payload sizes
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        let ty = &data[pos + 4..pos + 8];
        let end = pos.checked_add(12)?.checked_add(len)?;
        if end > data.len() {
            return None;
        }
        // a CRC that doesn't verify can't be regenerated, so leave the file alone
        let crc = u32::from_be_bytes(data[end - 4..end].try_into().ok()?);
        if crc != crc32(&data[pos + 4..end - 4]) {
            return None;
        }
        match ty {
            b"IHDR" => ihdr = Some(&data[pos + 8..end - 4]),
            b"IDAT" => match &mut idat {
                None => idat = Some((pos, end, vec![len as u32])),
                Some((_, e, sizes)) if *e == pos => {
                    *e = end;
                    sizes.push(len as u32);
                }
                Some(_) => return None, // IDATs not contiguous: unusual, skip
            },
            b"IEND" => break,
            _ => {}
        }
        pos = end;
    }
    let ihdr = ihdr?;
    let (idat_start, idat_end, sizes) = idat?;
    if ihdr.len() != 13 {
        return None;
    }
    let mut stream = Vec::new();
    {
        let mut p = idat_start;
        for &s in &sizes {
            stream.extend_from_slice(&data[p + 8..p + 8 + s as usize]);
            p += 12 + s as usize;
        }
    }
    // the zlib stream: header, deflate, Adler-32 — and nothing after it
    if stream.len() < 6 {
        return None;
    }
    let (cmf, flg) = (stream[0], stream[1]);
    if cmf & 0x0f != 8 || (cmf as u16 * 256 + flg as u16) % 31 != 0 || flg & 0x20 != 0 {
        return None;
    }
    let (s0, s1, method, raw) = try_stream(&stream, 2, (cmf >> 4) + 8)?;
    if s1 + 4 != stream.len() || u32::from_be_bytes(stream[s1..s1 + 4].try_into().ok()?) != adler32(&raw) {
        return None;
    }
    // what the scanlines are
    let w = u32::from_be_bytes(ihdr[0..4].try_into().ok()?) as usize;
    let h = u32::from_be_bytes(ihdr[4..8].try_into().ok()?) as usize;
    let (bit_depth, color, interlace) = (ihdr[8] as usize, ihdr[9], ihdr[12]);
    let chans = match color {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        _ => return None,
    };
    let rowbytes = (w * chans * bit_depth).div_ceil(8);
    let bpp = (chans * bit_depth).div_ceil(8);

    // emit: raw bytes up to the first IDAT, then the Idat piece wrapping the
    // zlib stream, whose deflate part wraps the unfiltered image
    virt.extend_from_slice(&data[..idat_start]);
    let idat_at = virt.len();
    virt.extend_from_slice(&stream[..s0]);
    let mut lay = None;
    let defl = emit(method, virt, |virt| {
        let defl_at = virt.len();
        if interlace == 0 && raw.len() == h * (rowbytes + 1) && w > 0 {
            if let Some((filters, pixels)) = unfilter(&raw, rowbytes, h, bpp) {
                virt.extend_from_slice(&filters);
                let pix_at = virt.len();
                virt.extend_from_slice(&pixels);
                // 8- and 16-bit greyscale and colour are numbers; palettes and
                // packed low bit depths are not, and go to the byte models
                if color != 3 && (bit_depth == 8 || bit_depth == 16) && h >= 2 && w * h >= 1024 {
                    let mut l = Layout {
                        kind: front::KIND_IMAGE,
                        off: 0,
                        count: w * h * chans,
                        width: (bit_depth / 8) as u8,
                        shift: 0,
                        flags: front::LAY_BE,
                        chans: chans as u8,
                        row: w * chans,
                        stride: rowbytes,
                    };
                    l.detect_shift(&pixels);
                    l.off = pix_at;
                    lay = Some(l);
                }
                let k = Kind::Unfilter { rowbytes, rows: h, bpp };
                return vec![Piece { start: defl_at, virt_len: virt.len() - defl_at, kind: k, children: Vec::new() }];
            }
        }
        virt.extend_from_slice(&raw);
        Vec::new()
    });
    lays.extend(lay);
    virt.extend_from_slice(&stream[s1..]);
    let idat = Piece { start: idat_at, virt_len: virt.len() - idat_at, kind: Kind::Idat(sizes), children: vec![defl] };
    virt.extend_from_slice(&data[idat_end..]);
    Some(vec![idat])
}

/// Walk a JPEG from its SOI: returns its length (through EOI) and the
/// entropy-coded segment of each scan, as (offset, length) — only for the
/// sequential Huffman JPEGs the model understands.
fn jpeg_scans(d: &[u8]) -> Option<(usize, Vec<(usize, usize)>)> {
    let mut p = 2;
    let mut scans = Vec::new();
    let mut sequential = false;
    loop {
        // markers may be preceded by fill bytes
        while *d.get(p)? == 0xff && *d.get(p + 1)? == 0xff {
            p += 1;
        }
        if *d.get(p)? != 0xff {
            return None;
        }
        let m = *d.get(p + 1)?;
        match m {
            0xd9 => return Some((p + 2, if sequential { scans } else { Vec::new() })),
            0xd8 | 0x01 | 0xd0..=0xd7 => {
                p += 2;
                continue;
            }
            _ => {}
        }
        let len = ((*d.get(p + 2)? as usize) << 8) | *d.get(p + 3)? as usize;
        if len < 2 {
            return None;
        }
        if m == 0xc0 || m == 0xc1 {
            sequential = true;
        }
        p += 2 + len;
        if m == 0xda {
            // entropy-coded data runs to the first marker that is neither
            // stuffing nor a restart
            let st = p;
            while !(*d.get(p)? == 0xff && !matches!(*d.get(p + 1)?, 0x00 | 0xd0..=0xd7)) {
                p += 1;
            }
            if p - st >= 64 {
                scans.push((st, p - st));
            }
        }
    }
}

/// Length of the PNG starting at `d` (through its IEND chunk), if it parses.
fn png_end(d: &[u8]) -> Option<usize> {
    let mut pos = 8;
    while pos + 12 <= d.len() {
        let len = u32::from_be_bytes(d[pos..pos + 4].try_into().ok()?) as usize;
        let end = pos.checked_add(12)?.checked_add(len)?;
        if end > d.len() {
            return None;
        }
        if &d[pos + 4..pos + 8] == b"IEND" {
            return Some(end);
        }
        pos = end;
    }
    None
}

#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
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

/// Undo PNG row filters: returns (filter type per row, raw pixel bytes).
fn unfilter(f: &[u8], rowbytes: usize, rows: usize, bpp: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut types = Vec::with_capacity(rows);
    let mut out = vec![0u8; rows * rowbytes];
    for y in 0..rows {
        let t = f[y * (rowbytes + 1)];
        if t > 4 {
            return None;
        }
        types.push(t);
        let src = &f[y * (rowbytes + 1) + 1..(y + 1) * (rowbytes + 1)];
        for x in 0..rowbytes {
            let a = if x >= bpp { out[y * rowbytes + x - bpp] } else { 0 };
            let b = if y > 0 { out[(y - 1) * rowbytes + x] } else { 0 };
            let c = if x >= bpp && y > 0 { out[(y - 1) * rowbytes + x - bpp] } else { 0 };
            let pred = match t {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                _ => paeth(a, b, c),
            };
            out[y * rowbytes + x] = src[x].wrapping_add(pred);
        }
    }
    Some((types, out))
}

fn refilter(types: &[u8], pix: &[u8], rowbytes: usize, bpp: usize) -> Vec<u8> {
    let rows = types.len();
    let mut out = Vec::with_capacity(rows * (rowbytes + 1));
    for y in 0..rows {
        let t = types[y];
        out.push(t);
        for x in 0..rowbytes {
            let a = if x >= bpp { pix[y * rowbytes + x - bpp] } else { 0 };
            let b = if y > 0 { pix[(y - 1) * rowbytes + x] } else { 0 };
            let c = if x >= bpp && y > 0 { pix[(y - 1) * rowbytes + x - bpp] } else { 0 };
            let pred = match t {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                _ => paeth(a, b, c),
            };
            out.push(pix[y * rowbytes + x].wrapping_sub(pred));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Rebuild
// ---------------------------------------------------------------------------

pub fn rebuild(virt: &[u8], pieces: &[Piece]) -> Option<Vec<u8>> {
    rebuild_span(virt, 0, virt.len(), pieces)
}

fn rebuild_span(virt: &[u8], lo: usize, hi: usize, pieces: &[Piece]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(hi - lo);
    let mut pos = lo;
    for p in pieces {
        let end = p.start.checked_add(p.virt_len)?;
        if p.start < pos || end > hi {
            return None;
        }
        out.extend_from_slice(&virt[pos..p.start]);
        let inner = rebuild_span(virt, p.start, end, &p.children)?;
        pos = end;
        match &p.kind {
            Kind::Deflate(params) => out.extend_from_slice(&deflate::deflate(&inner, *params)),
            Kind::Reflate { diff_len } => {
                let (diff, data) = inner.split_at_checked(*diff_len)?;
                out.extend_from_slice(&super::reflate::rebuild(data, diff)?);
            }
            Kind::Gif { lzw, width, height, interlaced } => {
                if inner.len() != width.checked_mul(*height)? {
                    return None;
                }
                let pixels = if *interlaced {
                    let mut v = vec![0u8; inner.len()];
                    for (k, &r) in super::gif::interlace_order(*height).iter().enumerate() {
                        v[k * width..(k + 1) * width].copy_from_slice(&inner[r * width..(r + 1) * width]);
                    }
                    v
                } else {
                    inner
                };
                out.extend_from_slice(&super::gif::blocks(&super::gif::encode(&pixels, lzw), lzw)?);
            }
            Kind::Idat(sizes) => {
                let mut at = 0usize;
                for &s in sizes {
                    let s = s as usize;
                    let payload = inner.get(at..at.checked_add(s)?)?;
                    out.extend_from_slice(&(s as u32).to_be_bytes());
                    let start = out.len();
                    out.extend_from_slice(b"IDAT");
                    out.extend_from_slice(payload);
                    let crc = crc32(&out[start..]);
                    out.extend_from_slice(&crc.to_be_bytes());
                    at += s;
                }
                if at != inner.len() {
                    return None;
                }
            }
            Kind::Unfilter { rowbytes, rows, bpp } => {
                if inner.len() != rows.checked_mul(rowbytes.checked_add(1)?)? {
                    return None;
                }
                out.extend_from_slice(&refilter(&inner[..*rows], &inner[*rows..], *rowbytes, *bpp));
            }
        }
    }
    out.extend_from_slice(&virt[pos..hi]);
    Some(out)
}

// ---------------------------------------------------------------------------
// Recipe serialisation
// ---------------------------------------------------------------------------

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

fn put_pieces(out: &mut Vec<u8>, ps: &[Piece], mut prev_end: usize) {
    put_var(out, ps.len() as u64);
    for p in ps {
        put_var(out, (p.start - prev_end) as u64);
        prev_end = p.start + p.virt_len;
        put_var(out, p.virt_len as u64);
        match &p.kind {
            Kind::Deflate(q) => {
                out.push(1);
                out.extend_from_slice(&[q.level, q.mem_level, q.wbits, q.strategy as u8]);
            }
            Kind::Idat(sizes) => {
                out.push(2);
                put_var(out, sizes.len() as u64);
                for &s in sizes {
                    put_var(out, s as u64);
                }
            }
            Kind::Reflate { diff_len } => {
                out.push(4);
                put_var(out, *diff_len as u64);
            }
            Kind::Gif { lzw, width, height, interlaced } => {
                out.push(5);
                out.push(lzw.min_size | (lzw.end_code as u8) << 4 | (*interlaced as u8) << 5);
                put_var(out, *width as u64);
                put_var(out, *height as u64);
                put_var(out, lzw.clears.len() as u64);
                let mut prev = 0;
                for &c in &lzw.clears {
                    put_var(out, (c - prev) as u64);
                    prev = c;
                }
                put_var(out, lzw.cuts.len() as u64);
                let mut prev = 0;
                for &(at, len) in &lzw.cuts {
                    put_var(out, (at - prev) as u64);
                    put_var(out, len as u64);
                    prev = at;
                }
                // sub-blocks are nearly always full: then only their count
                if lzw.blocks.iter().all(|&b| b == 255) {
                    out.push(0);
                    put_var(out, lzw.blocks.len() as u64);
                } else {
                    out.push(1);
                    put_var(out, lzw.blocks.len() as u64);
                    out.extend_from_slice(&lzw.blocks);
                }
                out.push(lzw.last_block);
            }
            Kind::Unfilter { rowbytes, rows, bpp } => {
                out.push(3);
                put_var(out, *rowbytes as u64);
                put_var(out, *rows as u64);
                put_var(out, *bpp as u64);
            }
        }
        put_pieces(out, &p.children, p.start);
    }
}

fn get_pieces(d: &[u8], p: &mut usize, depth: u32, mut prev_end: usize) -> Option<Vec<Piece>> {
    if depth > MAX_DEPTH + 2 {
        return None;
    }
    let n = get_var(d, p)? as usize;
    if n > d.len() {
        return None;
    }
    let mut ps = Vec::with_capacity(n);
    for _ in 0..n {
        let start = prev_end.checked_add(get_var(d, p)? as usize)?;
        let virt_len = get_var(d, p)? as usize;
        prev_end = start.checked_add(virt_len)?;
        let kind = match *d.get(*p)? {
            1 => {
                let b = d.get(*p + 1..*p + 5)?;
                *p += 5;
                let q = Params { level: b[0], mem_level: b[1], wbits: b[2], strategy: Strategy::from_u8(b[3])? };
                if !(1..=9).contains(&q.level) || !(1..=9).contains(&q.mem_level) || !(8..=15).contains(&q.wbits) {
                    return None;
                }
                Kind::Deflate(q)
            }
            2 => {
                *p += 1;
                let k = get_var(d, p)? as usize;
                if k > d.len() {
                    return None;
                }
                let mut sizes = Vec::with_capacity(k);
                for _ in 0..k {
                    sizes.push(u32::try_from(get_var(d, p)?).ok()?);
                }
                Kind::Idat(sizes)
            }
            4 => {
                *p += 1;
                Kind::Reflate { diff_len: get_var(d, p)? as usize }
            }
            5 => {
                let f = *d.get(*p + 1)?;
                *p += 2;
                let (width, height) = (get_var(d, p)? as usize, get_var(d, p)? as usize);
                let n = get_var(d, p)? as usize;
                if n > d.len() || width.checked_mul(height)? > 1 << 30 {
                    return None;
                }
                let mut clears = Vec::with_capacity(n);
                let mut acc = 0u32;
                for _ in 0..n {
                    acc = acc.checked_add(u32::try_from(get_var(d, p)?).ok()?)?;
                    clears.push(acc);
                }
                let nc = get_var(d, p)? as usize;
                if nc > d.len() {
                    return None;
                }
                let mut cuts = Vec::with_capacity(nc);
                let mut at = 0u32;
                for _ in 0..nc {
                    at = at.checked_add(u32::try_from(get_var(d, p)?).ok()?)?;
                    cuts.push((at, u16::try_from(get_var(d, p)?).ok()?));
                }
                let mode = *d.get(*p)?;
                *p += 1;
                let nb = get_var(d, p)? as usize;
                if nb > (1 << 30) / 255 {
                    return None;
                }
                let blocks = if mode == 0 {
                    vec![255u8; nb]
                } else {
                    let b = d.get(*p..*p + nb)?.to_vec();
                    *p += nb;
                    b
                };
                let last_block = *d.get(*p)?;
                *p += 1;
                let min_size = f & 15;
                if !(2..=8).contains(&min_size) {
                    return None;
                }
                let lzw = super::gif::Lzw { min_size, clears, cuts, end_code: f & 16 != 0, blocks, last_block };
                Kind::Gif { lzw, width, height, interlaced: f & 32 != 0 }
            }
            3 => {
                *p += 1;
                let rowbytes = get_var(d, p)? as usize;
                let rows = get_var(d, p)? as usize;
                let bpp = get_var(d, p)? as usize;
                if bpp == 0 || bpp > 8 || rowbytes.checked_add(1)?.checked_mul(rows)? > 1 << 34 {
                    return None;
                }
                Kind::Unfilter { rowbytes, rows, bpp }
            }
            _ => return None,
        };
        let children = get_pieces(d, p, depth + 1, start)?;
        ps.push(Piece { start, virt_len, kind, children });
    }
    Some(ps)
}

/// Serialise layouts and pieces.
pub fn recipe_bytes(pieces: &[Piece], layouts: &[Layout]) -> Vec<u8> {
    let mut out = Vec::new();
    put_var(&mut out, layouts.len() as u64);
    for l in layouts {
        out.extend_from_slice(&l.to_bytes());
    }
    put_pieces(&mut out, pieces, 0);
    out
}

pub fn parse_recipe(d: &[u8], virt_len: usize) -> Option<(Vec<Piece>, Vec<Layout>)> {
    let mut p = 0;
    let n = get_var(d, &mut p)? as usize;
    if n > d.len() {
        return None;
    }
    let mut lays = Vec::with_capacity(n);
    let mut prev_end = 0;
    for _ in 0..n {
        let l = Layout::from_bytes(d.get(p..)?, virt_len)?;
        // layouts must be in order and disjoint for the coder to walk them
        if l.off < prev_end {
            return None;
        }
        prev_end = l.end();
        lays.push(l);
        p += front::LAYOUT_LEN;
    }
    let pieces = get_pieces(d, &mut p, 0, 0)?;
    (p == d.len()).then_some((pieces, lays))
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

pub fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in d.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

pub fn crc32(d: &[u8]) -> u32 {
    use std::sync::OnceLock;
    static T: OnceLock<[u32; 256]> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, e) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for &x in d {
        c = t[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}
