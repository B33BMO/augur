//! Text model: words, the gaps between them, and the shape of the page.
//!
//! The main predictor's word contexts see the word being typed and the one or
//! two before it. Prose has far more structure than that: the punctuation and
//! spacing between words ("gap tokens") predict the next word's case and kind;
//! a word two or three back predicts this one even with a different word in
//! between; brackets and quotes open regions with their own vocabulary; a
//! column of a table or an indented block repeats what was above it; and a
//! paragraph tends to keep using the characters it started with. Each of
//! these is a context here, looked up per bit through a context bank.

use super::ctxbank::CtxBank;

const NT: usize = 24;
pub const TEXT_IN: usize = 2 * NT;
const TABLE_BITS: u32 = 22;

#[inline]
fn hstep(h: u32, c: u32) -> u32 {
    (h ^ c).wrapping_mul(0x0100_0193).rotate_left(5)
}

pub struct TextModel {
    c4: u32,
    word: [u32; 5],    // current word, then the four before it
    gap: [u32; 2],     // the token between words: now, and before the last word
    in_word: bool,
    letters: u32,      // hash of the last letters, ignoring everything else
    letter_ring: [u8; 6],
    mask: u32,         // 2-bit character classes, newest lowest
    brackets: [u8; 16],
    depth: usize,
    col: u32,
    line_start: usize,
    prev_line: usize,
    prev_len: u32,
    para_first: u32,   // first byte and first word of this paragraph
    para_word: u32,
    blank_run: u32,    // consecutive newlines
    sentence_start: bool,
    cap: u32,          // capitalisation of the current word: 0 none, 1 first, 2 all so far
    digits: u32,
    buf: Vec<u8>,      // recent bytes, for the line above
    texty: u32,        // decayed count of text-like bytes, x256
    pub on: bool,
    bank: CtxBank,
}

/// Fraction of text-like bytes (of a ~512-byte window) needed: 97%.
const TEXTY_MIN: u32 = 512 * 256 * 97 / 100;
const BUF: usize = 1 << 16;

impl TextModel {
    pub fn new() -> Self {
        Self {
            c4: 0,
            word: [0; 5],
            gap: [0; 2],
            in_word: false,
            letters: 0,
            letter_ring: [0; 6],
            mask: 0,
            brackets: [0; 16],
            depth: 0,
            col: 0,
            line_start: 0,
            prev_line: 0,
            prev_len: 0,
            para_first: 0,
            para_word: 0,
            blank_run: 0,
            sentence_start: true,
            cap: 0,
            digits: 0,
            buf: vec![0; BUF],
            texty: 0,
            on: false,
            bank: CtxBank::new(NT, TABLE_BITS),
        }
    }

    /// Advance past byte `b`, at stream position `pos` (the count of bytes before it).
    pub fn byte(&mut self, b: u8, pos: usize) {
        self.buf[pos & (BUF - 1)] = b;
        let textlike = matches!(b, b'\t' | b'\n' | b'\r' | 0x20..=0x7e | 0x80..=0xff);
        self.texty = self.texty - (self.texty >> 9) + if textlike { 256 } else { 0 };
        self.on = self.texty >= TEXTY_MIN;
        self.c4 = self.c4 << 8 | b as u32;
        let lower = b.to_ascii_lowercase();
        let is_alnum = b.is_ascii_alphanumeric() || b >= 0x80;
        let class = if b.is_ascii_alphabetic() || b >= 0x80 {
            0
        } else if b.is_ascii_digit() {
            1
        } else if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
            2
        } else {
            3
        };
        self.mask = self.mask << 2 | class;
        if is_alnum {
            if !self.in_word {
                // a word begins: the gap that preceded it is complete
                self.gap[1] = self.gap[0];
                self.cap = if b.is_ascii_uppercase() { 1 } else { 0 };
            } else if self.cap > 0 && !b.is_ascii_uppercase() {
                self.cap = 1;
            } else if self.cap == 1 && b.is_ascii_uppercase() && self.word[0] != 0 {
                self.cap = 2;
            }
            self.word[0] = hstep(self.word[0], lower as u32);
            self.in_word = true;
            if b.is_ascii_alphabetic() || b >= 0x80 {
                self.letter_ring.rotate_right(1);
                self.letter_ring[0] = lower;
                self.letters = self.letter_ring.iter().fold(0, |h, &c| hstep(h, c as u32));
            }
            self.digits = if b.is_ascii_digit() { self.digits + 1 } else { 0 };
        } else {
            if self.in_word {
                // a word ends
                for k in (1..5).rev() {
                    self.word[k] = self.word[k - 1];
                }
                if self.para_word == 0 {
                    self.para_word = self.word[1];
                }
                self.word[0] = 0;
                self.gap[0] = 0;
                self.in_word = false;
            }
            self.gap[0] = hstep(self.gap[0], b as u32);
            self.digits = 0;
        }
        match b {
            b'.' | b'!' | b'?' => self.sentence_start = true,
            _ if is_alnum => self.sentence_start = false,
            _ => {}
        }
        match b {
            b'(' | b'[' | b'{' | b'<' => {
                if self.depth < 16 {
                    self.brackets[self.depth] = b;
                }
                self.depth += 1;
            }
            b')' | b']' | b'}' | b'>' => self.depth = self.depth.saturating_sub(1),
            _ => {}
        }
        if b == b'\n' {
            self.blank_run += 1;
            if self.blank_run >= 2 {
                self.para_first = 0;
                self.para_word = 0;
            }
            self.prev_len = self.col;
            self.prev_line = self.line_start;
            self.line_start = pos + 1;
            self.col = 0;
        } else {
            if self.blank_run >= 2 || self.para_first == 0 {
                if self.para_first == 0 && b != b' ' && b != b'\r' {
                    self.para_first = b as u32 | 0x100;
                }
            }
            if b != b'\r' {
                self.blank_run = 0;
            }
            self.col += 1;
        }
        self.contexts(pos + 1);
    }

    fn contexts(&mut self, next_pos: usize) {
        let c1 = self.c4 & 0xff;
        let [w0, w1, w2, w3, _] = self.word;
        let [g0, g1] = self.gap;
        let col = self.col.min(255);
        // the byte at this column in the line above, if the line was long enough
        let above = if self.col < self.prev_len && next_pos - self.prev_line < BUF {
            self.buf[(self.prev_line + self.col as usize) & (BUF - 1)] as u32
        } else {
            0x100
        };
        let above2 = if self.col + 1 < self.prev_len { self.buf[(self.prev_line + self.col as usize + 1) & (BUF - 1)] as u32 } else { 0x100 };
        let open = if self.depth > 0 { self.brackets[(self.depth - 1).min(15)] as u32 } else { 0 };
        let h = |tag: u32, a: u32, b: u32, c: u32| {
            tag.wrapping_mul(0x9e37_79b1) ^ a.wrapping_mul(0x85eb_ca6b) ^ b.wrapping_mul(0xc2b2_ae35) ^ c.wrapping_mul(0x27d4_eb2f)
        };
        let state = self.sentence_start as u32 | self.cap << 1 | (self.in_word as u32) << 3;
        let base = [
            h(1, w0, g0, 0),
            h(2, c1, w0, g1),
            h(3, c1, g0, w1),
            h(4, w0, w1, 0),
            h(5, w0, w1, w2),
            h(6, g0, w1, g1 ^ w2.rotate_left(7)),
            h(7, w0, c1, w2),
            h(8, w0, c1, w3),
            h(9, open, w0, self.depth.min(7) as u32),
            h(10, open, c1, self.depth.min(7) as u32),
            h(11, col << 8 | c1, 0, 0),
            h(12, above, c1, col.min(31)),
            h(13, above, above2, c1),
            h(14, self.para_first, c1, 0),
            h(15, self.para_word, w0, 0),
            h(16, self.mask & 0xffff, 0, 0),
            h(17, self.mask & 0xff, c1, 0),
            h(18, self.letters, 0, 0),
            h(19, state, w0, c1),
            h(20, state, w1, g0),
            h(21, w0, self.prev_len.min(127), col.min(15)),
            h(22, w2, w0, g0),
            h(23, self.c4 & 0xffff, w0, 0),
            h(24, self.digits.min(15), c1, self.c4 >> 8 & 0xff),
        ];
        self.bank.base.copy_from_slice(&base);
    }

    /// Mixer selector: where in the text we are.
    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        let state = self.sentence_start as u32 | self.cap << 1 | (self.in_word as u32) << 3;
        ((state << 4 | (self.mask & 15)) << 3 | bitpos) as usize
    }

    #[inline]
    pub fn inputs(&mut self, c0: u32, st: &mut [i32]) {
        self.bank.inputs(c0, st);
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        self.bank.update(bit);
    }
}
