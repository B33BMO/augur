//! A bank of hashed bit-level contexts.
//!
//! The byte models in the main predictor live in checksummed nibble slots;
//! the add-on models (x86, text) need something lighter: a set of context
//! hashes fixed for a byte, each looked up per bit together with the bits of
//! the byte coded so far, giving the mixer two inputs per context — a direct
//! counter, and a bit history read through a learned map.

use super::{sm_init, sm_update, state_tab, stretch_tab, StateTab, RATE_TAB};

pub struct CtxBank {
    n: usize,
    bits: u32,
    table: Vec<u32>,
    bh: Vec<u8>,
    sm: Vec<u32>,
    stt: &'static StateTab,
    ix: Vec<usize>,
    pub base: Vec<u32>,
    fresh: bool,
}

impl CtxBank {
    pub fn new(n: usize, bits: u32) -> Self {
        Self {
            n,
            bits,
            table: vec![(1 << 21) << 10; 1 << bits],
            bh: vec![0; 1 << bits],
            sm: (0..n).flat_map(|_| sm_init(state_tab())).collect(),
            stt: state_tab(),
            ix: vec![0; n],
            base: vec![0; n],
            fresh: false,
        }
    }

    /// Look up every context for the bit after partial byte `c0`.
    #[inline]
    pub fn inputs(&mut self, c0: u32, st: &mut [i32]) {
        let stab = stretch_tab();
        let n = self.n;
        for i in 0..n {
            let k = self.base[i] ^ c0.wrapping_mul(0x6c8e_9cf5) ^ (i as u32) << 27;
            let ix = (k.wrapping_mul(0x2545_f491) >> (32 - self.bits)) as usize;
            self.ix[i] = ix;
            st[i] = stab[(self.table[ix] >> 20) as usize];
            st[n + i] = stab[(self.sm[(i << 8) | self.bh[ix] as usize] >> 20) as usize];
        }
        self.fresh = true;
    }

    #[inline]
    pub fn update(&mut self, bit: u32) {
        if !self.fresh {
            return;
        }
        self.fresh = false;
        for i in 0..self.n {
            let ix = self.ix[i];
            let v = &mut self.table[ix];
            let cnt = *v & 1023;
            let p22 = (*v >> 10) as i32;
            let err = (((bit as i32) << 22) - p22) as i64;
            let p22 = (p22 + ((err * RATE_TAB[cnt as usize] as i64) >> 16) as i32).clamp(0, (1 << 22) - 1) as u32;
            *v = (p22 << 10) | if cnt < 255 { cnt + 1 } else { cnt };
            let hs = self.bh[ix] as usize;
            let c = &mut self.sm[(i << 8) | hs];
            *c = sm_update(*c, bit);
            self.bh[ix] = self.stt.next[hs][bit as usize];
        }
    }
}
