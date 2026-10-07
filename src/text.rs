//! Text model: words, the tokens between them, and the shape of the page.
//!
//! The order-n contexts see bytes; prose is made of words. This model follows
//! the text as words and gap tokens (the punctuation and spacing between
//! words), and hands the main predictor a set of contexts built from them —
//! which flow through the same checksummed slots, bit histories and byte
//! histories as the order-n contexts. Measured on paq8px, this kind of word
//! modelling is worth 7% on prose by itself; its ideas are followed here:
//!
//! - word history shifts on punctuation too, so "word, word" and "word word"
//!   are different contexts, and sentence ends clear it;
//! - the last five letters or digits, ignoring everything else;
//! - how long ago the current word last appeared;
//! - sections introduced by "keyword:" or "keyword=";
//! - expressions (letters separated by single spaces), word morphology
//!   (vowel/consonant types of the last letters), character groups;
//! - line structure: column, the character above, paragraph and line starts.

/// Contexts this model contributes to the main predictor's slots.
pub const NTXT: usize = 32;

const WPOS_BITS: u32 = 20;
const BUF: usize = 1 << 16;

#[inline]
fn comb(h: u64, c: u64) -> u64 {
    (h.wrapping_add(c + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23)
}

#[inline]
fn hh(tag: u64, xs: &[u64]) -> u32 {
    let mut h = tag.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    for &x in xs {
        h = (h ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    (h >> 32) as u32 ^ h as u32
}

pub struct TextModel {
    c4: u32,
    c: u8,  // last byte, lower-cased
    pc: u8, // the one before
    is_letter: bool,
    pc_letter: bool,
    ppc_letter: bool,
    newline: bool,
    pc_newline: bool,
    word: [u64; 5],
    word_len: u32,
    gap: [u64; 2],
    text0: u64, // last five letters/digits
    keyword: u64,
    first_word: u64,
    expr: [u64; 5],
    expr_len: u32,
    expr_chars: u32,
    mask2: u32, // types of the last letters: vowel, e, consonant, th, digit, punctuation
    groups: u64,
    opened: u8,
    last_upper: u32,
    // word recency
    wpos: Vec<u32>,
    wchk: Vec<u16>,
    // lines
    col: u32,
    line_start: usize,
    prev_line: usize,
    prev_len: u32,
    first_char: u32,
    para_start: usize,
    buf: Vec<u8>,
    texty: u32,
    pub on: bool,
    pub ctx: [u32; NTXT],
}

/// Fraction of text-like bytes (of a ~512-byte window) needed: 97%.
const TEXTY_MIN: u32 = 512 * 256 * 97 / 100;

impl TextModel {
    pub fn new() -> Self {
        Self {
            c4: 0,
            c: 0,
            pc: 0,
            is_letter: false,
            pc_letter: false,
            ppc_letter: false,
            newline: false,
            pc_newline: false,
            word: [0; 5],
            word_len: 0,
            gap: [0; 2],
            text0: 0,
            keyword: 0,
            first_word: 0,
            expr: [0; 5],
            expr_len: 0,
            expr_chars: 0,
            mask2: 0,
            groups: 0,
            opened: 0,
            last_upper: 64,
            wpos: vec![0; 1 << WPOS_BITS],
            wchk: vec![0; 1 << WPOS_BITS],
            col: 0,
            line_start: 0,
            prev_line: 0,
            prev_len: 0,
            first_char: 0,
            para_start: 0,
            buf: vec![0; BUF],
            texty: 0,
            on: false,
            ctx: [0; NTXT],
        }
    }

    fn wslot(h: u64) -> (usize, u16) {
        ((h >> (64 - WPOS_BITS)) as usize, (h >> 8) as u16 | 1)
    }

    /// Advance past byte `b`, which sits at stream position `pos`.
    pub fn byte(&mut self, b: u8, pos: usize) {
        self.buf[pos & (BUF - 1)] = b;
        let textlike = matches!(b, b'\t' | b'\n' | b'\r' | 0x20..=0x7e | 0x80..=0xff);
        self.texty = self.texty - (self.texty >> 9) + if textlike { 256 } else { 0 };
        self.on = self.texty >= TEXTY_MIN;

        self.ppc_letter = self.pc_letter;
        self.pc_letter = self.is_letter;
        self.pc_newline = self.newline;
        let ppc = self.pc;
        self.pc = self.c;
        self.c4 = self.c4 << 8 | b as u32;
        let mut c = b;
        if b.is_ascii_uppercase() {
            c = b.to_ascii_lowercase();
            self.last_upper = 0;
        }
        self.c = c;
        self.last_upper = (self.last_upper + 1).min(64);
        self.is_letter = c.is_ascii_lowercase() || b >= 0x80;
        let is_number = c.is_ascii_digit() || (self.pc.is_ascii_digit() && c == b'.');
        self.newline = b == b'\n';
        self.mask2 <<= 8;

        if self.is_letter || is_number {
            if self.word_len == 0 {
                // a word starts: the gap before it is complete
                self.gap[1] = self.gap[0];
                if self.pc == b'"' || (self.pc == b'\'' && !self.ppc_letter) {
                    self.opened = self.pc;
                }
                self.gap[0] = 0;
                self.mask2 = 0;
            }
            self.word[0] = comb(self.word[0], c as u64);
            self.text0 = (self.text0 << 8 | c as u64) & 0xff_ffff_ffff;
            self.word_len = (self.word_len + 1).min(45);
            if self.is_letter {
                self.mask2 |= match c {
                    b'e' => b'e' as u32,
                    b'a' | b'i' | b'o' | b'u' => b'a' as u32,
                    b'y' => b'y' as u32,
                    b'h' if self.pc == b't' => {
                        self.mask2 = (self.mask2 >> 8) & 0x00ff_ff00;
                        b't' as u32
                    }
                    b'b'..=b'z' => b'b' as u32,
                    _ => 128,
                };
            } else {
                self.mask2 |= match c {
                    b'.' => b'.' as u32,
                    b'0' => b'0' as u32,
                    _ => b'1' as u32,
                };
            }
        } else {
            self.gap[0] = comb(self.gap[0], if self.newline { b' ' as u64 } else { b as u64 });
            if c == b'?' || self.pc == b'!' || self.pc == b'.' {
                // a sentence ends: what came before says little now
                self.word[1..].fill(0);
                self.gap[1] = 0;
            } else if c == self.pc || ((c == b' ' || self.newline) && (self.pc == b' ' || self.pc_newline)) {
                // repeats and runs of whitespace don't shift the words
            } else {
                for k in (1..5).rev() {
                    self.word[k] = self.word[k - 1];
                }
            }
            if self.word_len != 0 {
                // the word just ended: note where, for "how long since"
                let (s, chk) = Self::wslot(self.word[0]);
                self.wpos[s] = pos as u32;
                self.wchk[s] = chk;
                if c == b':' || c == b'=' {
                    self.keyword = self.word[0];
                }
                if self.first_word == 0 {
                    self.first_word = self.word[0];
                }
                self.word[0] = 0;
                self.word_len = 0;
                self.mask2 = 0;
            }
            self.mask2 |= match b {
                b'.' | b'!' | b'?' => b'!' as u32,
                b',' | b';' | b':' => b',' as u32,
                b'(' | b'{' | b'[' | b'<' => {
                    self.opened = b;
                    b'(' as u32
                }
                b')' | b'}' | b']' | b'>' => {
                    self.opened = 0;
                    b')' as u32
                }
                b'"' | b'\'' => {
                    self.opened = 0;
                    b as u32
                }
                _ => b as u32,
            };
        }
        let g = match b {
            0x80..=0xff => match b {
                _ if b & 0xf8 == 0xf0 => 1,
                _ if b & 0xf0 == 0xe0 => 2,
                _ if b & 0xe0 == 0xc0 => 3,
                _ if b & 0xc0 == 0x80 => 4,
                _ => b & 0xf0,
            },
            b'0'..=b'9' => b'0',
            b'a'..=b'z' => b'a',
            b'A'..=b'Z' => b'A',
            0..=31 if !self.newline => 6,
            _ => b,
        };
        self.groups = self.groups << 8 | g as u64;
        // expressions: letters separated by single spaces
        if self.is_letter {
            self.expr_chars = self.expr_chars << 8 | c as u32;
            self.expr[0] = comb(self.expr[0], c as u64);
            self.expr_len = (self.expr_len + 1).min(45);
        } else {
            self.expr_chars = 0;
            self.expr_len = 0;
            if (c == b' ' || self.newline) && (self.pc_letter || self.pc == b'\'' || self.pc == b'"') {
                for k in (1..5).rev() {
                    self.expr[k] = self.expr[k - 1];
                }
                self.expr[0] = 0;
            } else if c == b'\'' || c == b'"' || (self.newline && self.pc == b' ') || (c == b' ' && self.pc_newline) {
            } else {
                self.expr = [0; 5];
            }
        }
        let _ = ppc;
        // lines and paragraphs
        if self.newline {
            if pos + 1 - self.line_start <= 1 {
                self.para_start = pos + 1; // a blank line
                self.first_word = 0;
            }
            self.prev_len = self.col;
            self.prev_line = self.line_start;
            self.line_start = pos + 1;
            self.col = 0;
            self.first_char = 0;
        } else {
            if self.col == 0 {
                self.first_char = b as u32 | 0x100;
            }
            self.col += 1;
        }
        self.contexts(pos + 1);
    }

    fn contexts(&mut self, next_pos: usize) {
        let c = self.c as u64;
        let c1 = (self.c4 & 0xff) as u64;
        let [w0, w1, w2, w3, w4] = self.word;
        let [g0, g1] = self.gap;
        let (s, chk) = Self::wslot(w0);
        let last = if self.wchk[s] == chk { self.wpos[s] } else { 0 };
        let dist = if last == 0 { 0 } else { (64 - ((next_pos as u64 - last as u64 + 120).leading_zeros() as u64)).min(20) };
        let may_end = (last != 0) as u64;
        let caps = ((self.c4 >> 8 & 0xff) as u8).is_ascii_uppercase() && (self.c4 as u8).is_ascii_uppercase();
        let wme = may_end << 1 | caps as u64;
        let wl = (self.word_len as u64).min(6) << 2 | wme;
        let col = self.col.min(255) as u64;
        let above = if (self.col) < self.prev_len { self.buf[(self.prev_line + self.col as usize) & (BUF - 1)] as u64 } else { 0x100 };
        let above2 = if self.col + 1 < self.prev_len { self.buf[(self.prev_line + self.col as usize + 1) & (BUF - 1)] as u64 } else { 0x100 };
        let el = self.expr_len as u64;
        let ec = self.expr_chars as u64;
        self.ctx = [
            hh(1, &[self.text0]),
            hh(2, &[self.expr[0], self.expr[1], self.expr[2], self.expr[3], self.expr[4]]),
            hh(3, &[self.expr[0], self.expr[1], self.expr[2]]),
            hh(4, &[g0, self.keyword]),
            hh(5, &[w0, c, self.keyword]),
            hh(6, &[w0, dist]),
            hh(7, &[w1, g0, dist]),
            hh(8, &[(next_pos >> 10) as u64, w0]),
            hh(9, &[wl, self.mask2 as u64]),
            hh(10, &[el.min(4) << 2 | wme, if el >= 2 { ec & 0xffff } else { c }]),
            hh(11, &[el.min(6) << 2 | wme, if el >= 3 { ec & 0xff_ffff } else { 0 }]),
            hh(12, &[w0, g0]),
            hh(13, &[c, w0, g1]),
            hh(14, &[c, g0, w1]),
            hh(15, &[w0, w1]),
            hh(16, &[w0, w1, w2]),
            hh(17, &[g0, w1, g1, w2]),
            hh(18, &[w0, w1, g1, w2]),
            hh(19, &[w0, w1, g1]),
            hh(20, &[w0, c1, w2]),
            hh(21, &[w0, c1, w3]),
            hh(22, &[w0, c1, w1, w4]),
            hh(23, &[self.opened as u64, wl, (dist != 0) as u64]),
            hh(24, &[self.opened as u64, w0]),
            hh(25, &[self.groups & 0xffff_ffff]),
            hh(26, &[self.groups & 0xff_ffff, c]),
            hh(27, &[col << 8 | c1, (self.c4 >> 8 & 0xff) as u64]),
            hh(28, &[above, above2, c1]),
            hh(29, &[col, self.prev_len.min(255) as u64, above]),
            hh(30, &[(self.para_start as u64) << 8 | c1]),
            hh(31, &[self.first_word, c]),
            hh(32, &[col << 8 | self.first_char as u64, (self.last_upper < self.col) as u64, self.groups & 0xff]),
        ];
    }

    /// Mixer selector: where in the text we are.
    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        let state = (self.word_len > 0) as u32 | ((self.last_upper < 2) as u32) << 1 | ((self.mask2 & 0xff) == b'!' as u32) as u32 * 4;
        ((state << 5 | (self.groups & 0x1f) as u32) << 3 | bitpos) as usize & 0x1ff
    }
}
