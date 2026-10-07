//! x86 machine-code model.
//!
//! An executable's bytes are instructions, and an instruction's bytes play
//! different roles: prefix, opcode, ModRM, SIB, displacement, immediate. A byte
//! model sees `8B 45 F8` as three unrelated bytes; this one knows the second is
//! the ModRM of a MOV and the third an 8-bit displacement off EBP, and that
//! displacements after `8B 45` are small negative stack offsets. It follows the
//! instruction stream as it is coded, and hands the mixer contexts made of each
//! byte's role, its instruction's opcode and ModRM, and the instructions
//! before it.
//!
//! The decoder tables below were written from the x86 encoding rules (which
//! opcodes take a ModRM byte, how wide their immediates are). The decoder
//! need not be perfect: data mixed into code throws it out of step for a few
//! bytes, after which it falls back into step, and both sides make the same
//! mistakes.

use super::ctxbank::CtxBank;

/// Hashed contexts per bit.
const NX: usize = 17;
/// Mixer inputs: a counter and a bit history per context.
pub const X86_IN: usize = 2 * NX;
const TABLE_BITS: u32 = 22;
/// Consecutive valid instructions before the model trusts that it is in code.
const MIN_VALID: u32 = 8;
/// Zero-byte density (x256 of a 512-byte window) below which it isn't code: 2%.
const ZERO_MIN: u32 = 512 * 256 / 50;

// Operand encoding of an opcode, packed in a byte:
//   bit 0: has a ModRM byte; bits 1-3: immediate kind; bit 4: invalid
const M: u8 = 1; // ModRM
const I8: u8 = 2; // 8-bit immediate
const IZ: u8 = 4; // 16/32-bit immediate (operand size)
const I16: u8 = 6; // 16-bit immediate
const R8: u8 = 8; // 8-bit relative branch
const RZ: u8 = 10; // 16/32-bit relative branch
const AD: u8 = 12; // absolute address (moffs)
const FAR: u8 = 14; // far pointer: 32-bit offset + 16-bit selector
const ENTER: u8 = 16; // imm16 + imm8
const BAD: u8 = 0x80;

/// Kind of the operand bytes for an opcode's immediate field.
fn imm_kind(f: u8) -> u8 {
    f & 0x1e
}

const fn one_byte() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 0x40 {
        // the eight ALU groups: r/m forms, then AL/eAX immediates; x6/x7 are
        // segment pushes or prefixes or BCD adjusts, none with operands
        t[i] = match i & 7 {
            0..=3 => M,
            4 => I8,
            5 => IZ,
            _ => 0,
        };
        i += 1;
    }
    t[0x0f] = 0; // escape, handled apart
    t[0x62] = M;
    t[0x63] = M;
    t[0x68] = IZ;
    t[0x69] = M | IZ;
    t[0x6a] = I8;
    t[0x6b] = M | I8;
    i = 0x70;
    while i < 0x80 {
        t[i] = R8;
        i += 1;
    }
    t[0x80] = M | I8;
    t[0x81] = M | IZ;
    t[0x82] = M | I8;
    t[0x83] = M | I8;
    i = 0x84;
    while i < 0x90 {
        t[i] = M;
        i += 1;
    }
    t[0x9a] = FAR;
    t[0xa0] = AD;
    t[0xa1] = AD;
    t[0xa2] = AD;
    t[0xa3] = AD;
    t[0xa8] = I8;
    t[0xa9] = IZ;
    i = 0xb0;
    while i < 0xb8 {
        t[i] = I8;
        i += 1;
    }
    while i < 0xc0 {
        t[i] = IZ;
        i += 1;
    }
    t[0xc0] = M | I8;
    t[0xc1] = M | I8;
    t[0xc2] = I16;
    t[0xc4] = M;
    t[0xc5] = M;
    t[0xc6] = M | I8;
    t[0xc7] = M | IZ;
    t[0xc8] = ENTER;
    t[0xca] = I16;
    t[0xcd] = I8;
    i = 0xd0;
    while i < 0xd4 {
        t[i] = M;
        i += 1;
    }
    t[0xd4] = I8;
    t[0xd5] = I8;
    i = 0xd8;
    while i < 0xe0 {
        t[i] = M; // x87
        i += 1;
    }
    i = 0xe0;
    while i < 0xe4 {
        t[i] = R8;
        i += 1;
    }
    while i < 0xe8 {
        t[i] = I8; // in/out imm8
        i += 1;
    }
    t[0xe8] = RZ;
    t[0xe9] = RZ;
    t[0xea] = FAR;
    t[0xeb] = R8;
    t[0xf6] = M; // test r/m8, imm8 when reg is 0 or 1: fixed up while decoding
    t[0xf7] = M;
    t[0xfe] = M;
    t[0xff] = M;
    t
}

const fn two_byte() -> [u8; 256] {
    let mut t = [M; 256];
    let none: [usize; 22] = [
        0x05, 0x06, 0x07, 0x08, 0x09, 0x0b, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x37, 0x77, 0xa0, 0xa1, 0xa2, 0xa8, 0xa9,
        0xaa, 0x0e, 0x04,
    ];
    let mut i = 0;
    while i < none.len() {
        t[none[i]] = 0;
        i += 1;
    }
    let bad: [usize; 9] = [0x0a, 0x0c, 0x24, 0x25, 0x26, 0x27, 0x36, 0x39, 0xff];
    i = 0;
    while i < bad.len() {
        t[bad[i]] = BAD;
        i += 1;
    }
    i = 0x3b;
    while i < 0x40 {
        t[i] = BAD;
        i += 1;
    }
    i = 0x80;
    while i < 0x90 {
        t[i] = RZ; // jcc rel32
        i += 1;
    }
    i = 0xc8;
    while i < 0xd0 {
        t[i] = 0; // bswap
        i += 1;
    }
    let m_i8: [usize; 10] = [0x70, 0x71, 0x72, 0x73, 0xa4, 0xac, 0xba, 0xc2, 0xc4, 0xc5];
    i = 0;
    while i < m_i8.len() {
        t[m_i8[i]] = M | I8;
        i += 1;
    }
    t[0xc6] = M | I8;
    t
}

static ONE: [u8; 256] = one_byte();
static TWO: [u8; 256] = two_byte();

/// What the next byte is.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    Opcode,    // first byte of an instruction (or a prefix)
    Opcode2,   // after 0F
    Opcode3,   // after 0F 38 / 0F 3A
    ModRm,
    Sib,
    Disp,      // displacement byte
    Imm,       // immediate byte
    Rel,       // relative branch target byte
}

impl Role {
    fn id(self) -> u32 {
        self as u32
    }
}

pub struct X86Model {
    // decoder state
    role: Role,
    left: u32,     // bytes left in the current displacement/immediate/relative field
    idx: u32,      // byte index within it
    prefixes: u32, // prefixes seen for this instruction (66, 67, F2/F3, segment, REX)
    opsize16: bool,
    op: u32,       // opcode (0F xx -> 0x100 | xx, 0F 38/3A xx -> 0x200/0x300 | xx)
    flags: u8,
    modrm: u32,
    sib: u32,
    pending_imm: u8, // immediate kind still to read after ModRM/SIB/displacement
    field: u32,      // bytes of the current multi-byte field so far
    hist: [u32; 4],  // opcodes of the last instructions
    last_len: u32,   // length of the previous instruction
    cur_len: u32,
    valid_run: u32,  // consecutive instructions decoded without error
    zeros: u32,      // decayed count of recent zero bytes, x256
    last_field: Vec<u32>, // per opcode (and ModRM mod/rm): the operand field it carried last
    state_hist: [u32; 64], // per (role, byte index): the last four bytes seen there
    pub on: bool,
    bank: CtxBank,
}

impl X86Model {
    pub fn new() -> Self {
        Self {
            role: Role::Opcode,
            left: 0,
            idx: 0,
            prefixes: 0,
            opsize16: false,
            op: 0,
            flags: 0,
            modrm: 0,
            sib: 0,
            pending_imm: 0,
            field: 0,
            hist: [0; 4],
            last_len: 0,
            cur_len: 0,
            valid_run: 0,
            zeros: 0,
            last_field: vec![0; 1 << 14],
            state_hist: [0; 64],
            on: false,
            bank: CtxBank::new(NX, TABLE_BITS),
        }
    }

    fn imm_bytes(&self, kind: u8) -> u32 {
        let z = if self.opsize16 { 2 } else { 4 };
        match kind {
            I8 | R8 => 1,
            IZ | RZ => z,
            I16 => 2,
            AD => 4,
            FAR => 6,
            ENTER => 3,
            _ => 0,
        }
    }

    fn end_instruction(&mut self, ok: bool) {
        self.hist = [self.op, self.hist[0], self.hist[1], self.hist[2]];
        self.last_len = self.cur_len;
        self.cur_len = 0;
        self.valid_run = if ok { self.valid_run + 1 } else { 0 };
        // decodes cleanly, and has the zero bytes displacements and
        // immediates are full of — which text, even UTF-8, never has
        self.on = self.valid_run >= MIN_VALID && self.zeros >= ZERO_MIN;
        self.role = Role::Opcode;
        self.prefixes = 0;
        self.opsize16 = false;
        self.modrm = 0;
        self.sib = 0;
        self.pending_imm = 0;
    }

    /// Move to the operand field `kind` (or finish the instruction).
    fn start_field(&mut self, role: Role, n: u32) {
        if n == 0 {
            self.end_instruction(true);
        } else {
            self.role = role;
            self.left = n;
            self.idx = 0;
            self.field = 0;
        }
    }

    fn after_modrm(&mut self) {
        let (md, rm) = (self.modrm >> 6, self.modrm & 7);
        let disp = match md {
            0 if rm == 5 => 4,
            0 if rm == 4 && self.sib & 7 == 5 => 4,
            1 => 1,
            2 => 4,
            _ => 0,
        };
        if disp > 0 {
            self.start_field(Role::Disp, disp);
        } else {
            self.after_disp();
        }
    }

    fn after_disp(&mut self) {
        let k = self.pending_imm;
        let n = self.imm_bytes(k);
        let role = if k == R8 || k == RZ { Role::Rel } else { Role::Imm };
        self.start_field(role, n);
    }

    fn opcode_flags(&mut self, f: u8) {
        self.flags = f;
        if f & BAD != 0 {
            self.end_instruction(false);
            return;
        }
        self.pending_imm = imm_kind(f);
        if f & M != 0 {
            self.role = Role::ModRm;
        } else {
            self.after_disp();
        }
    }

    /// Advance past one coded byte.
    pub fn byte(&mut self, b: u8) {
        self.cur_len += 1;
        // zero-byte density over roughly the last 512 bytes
        self.zeros = self.zeros - (self.zeros >> 9) + if b == 0 { 256 } else { 0 };
        let b32 = b as u32;
        match self.role {
            Role::Opcode => match b {
                0x66 => {
                    self.opsize16 = true;
                    self.prefixes |= 1;
                }
                0x67 => self.prefixes |= 2,
                0xf2 | 0xf3 => self.prefixes |= 4,
                0xf0 => self.prefixes |= 8,
                0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 => self.prefixes |= 16,
                0x0f => self.role = Role::Opcode2,
                _ => {
                    self.op = b32;
                    self.opcode_flags(ONE[b as usize]);
                }
            },
            Role::Opcode2 => {
                if b == 0x38 || b == 0x3a {
                    self.op = if b == 0x38 { 0x200 } else { 0x300 };
                    self.role = Role::Opcode3;
                } else {
                    self.op = 0x100 | b32;
                    self.opcode_flags(TWO[b as usize]);
                }
            }
            Role::Opcode3 => {
                let imm = if self.op == 0x300 { I8 } else { 0 };
                self.op |= b32;
                self.opcode_flags(M | imm);
            }
            Role::ModRm => {
                self.modrm = b32;
                // F6/F7 /0 and /1 (TEST) carry an immediate the table can't show
                if (self.op == 0xf6 || self.op == 0xf7) && (b32 >> 3) & 6 == 0 {
                    self.pending_imm = if self.op == 0xf6 { I8 } else { IZ };
                }
                if b32 >> 6 != 3 && b32 & 7 == 4 {
                    self.role = Role::Sib;
                } else {
                    self.after_modrm();
                }
            }
            Role::Sib => {
                self.sib = b32;
                self.after_modrm();
            }
            Role::Disp | Role::Imm | Role::Rel => {
                let sh = &mut self.state_hist[(self.role.id() as usize) << 3 | self.idx.min(7) as usize];
                *sh = *sh << 8 | b32;
                self.field |= b32 << (8 * self.idx);
                self.idx += 1;
                self.left -= 1;
                if self.left == 0 {
                    let k = self.field_key();
                    self.last_field[k] = self.field;
                    if self.role == Role::Disp {
                        self.after_disp();
                    } else {
                        self.end_instruction(true);
                    }
                }
            }
        }
        if self.cur_len > 15 {
            self.end_instruction(false);
        }
        self.contexts();
    }

    /// Which operand field this is, for its history: opcode, ModRM mod/rm,
    /// and whether it is the displacement or the immediate.
    fn field_key(&self) -> usize {
        ((self.op & 0x3ff) << 4 ^ (self.modrm & 0xc7) << 1 ^ (self.role == Role::Disp) as u32) as usize & ((1 << 14) - 1)
    }

    /// Contexts for the next byte.
    fn contexts(&mut self) {
        let r = self.role.id();
        let h = |tag: u32, a: u32, b: u32, c: u32| {
            (tag.wrapping_mul(0x9e37_79b1) ^ a.wrapping_mul(0x85eb_ca6b) ^ b.wrapping_mul(0xc2b2_ae35) ^ c.wrapping_mul(0x27d4_eb2f))
                .wrapping_add(r.wrapping_mul(0x1656_67b1))
        };
        let op = self.op;
        let mm = self.modrm;
        let pos = self.idx;
        let f = self.field;
        let [h1, h2, h3, _] = self.hist;
        let base = [
            h(1, op, pos, 0),
            h(2, op, mm, pos),
            h(3, h1, op, pos),
            h(4, h1, h2, h3 ^ op << 12),
            h(5, op, mm & 0xc7, f),
            h(6, op, self.prefixes, self.pending_imm as u32),
            h(7, h1, pos, mm >> 6),
            h(8, op, mm, self.sib),
            h(9, self.last_len, op, pos),
            h(10, if r == Role::Opcode.id() { h1 } else { op }, h2, 0),
            h(11, op, f, pos),
            h(12, mm >> 3 & 7, pos, op),
            h(13, h1 << 12 ^ op, mm, f & 0xff),
            h(14, r, pos, 0),
            {
                let in_field = matches!(self.role, Role::Disp | Role::Imm | Role::Rel);
                let last = if in_field { self.last_field[self.field_key()] } else { 0 };
                h(15, op, pos, (last >> (8 * pos.min(3))) & 0xff | (in_field as u32) << 8)
            },
            {
                let sh = self.state_hist[(r as usize) << 3 | pos.min(7) as usize];
                h(16, r, pos, sh & 0xffff)
            },
            {
                let in_field = matches!(self.role, Role::Disp | Role::Imm | Role::Rel);
                let last = if in_field { self.last_field[self.field_key()] } else { 0 };
                // the whole remembered field, against what is coded of this one
                h(17, op, last ^ f, pos)
            },
        ];
        self.bank.base.copy_from_slice(&base);
    }

    pub fn mixer_sel(&self, bitpos: u32) -> usize {
        ((self.role.id() << 6 | self.idx.min(7) << 3 | bitpos) & 1023) as usize
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
