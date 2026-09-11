//! Bitstream reader.
//!
//! This is a simplified but semantically equivalent translation of upstream's
//! `GetBitContext` (MSB-first bit order).

use crate::error::{DecoderError, Result};

#[derive(Clone)]
pub struct GetBitContext<'a> {
    buf: &'a [u8],
    size_in_bits: usize,
    bit_pos: usize,
}

impl<'a> GetBitContext<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            size_in_bits: buf.len() * 8,
            bit_pos: 0,
        }
    }

    pub fn new_bits(buf: &'a [u8], size_in_bits: usize) -> Result<Self> {
        if size_in_bits > buf.len() * 8 {
            return Err(DecoderError::InvalidData("bitstream size exceeds buffer".into()));
        }
        Ok(Self { buf, size_in_bits, bit_pos: 0 })
    }

    pub fn new_window(buf: &'a [u8], bit_pos: usize, size_in_bits: usize) -> Result<Self> {
        if bit_pos > size_in_bits || size_in_bits > buf.len() * 8 {
            return Err(DecoderError::InvalidData("invalid bitstream window".into()));
        }
        Ok(Self { buf, size_in_bits, bit_pos })
    }

    #[inline]
    pub fn bits_left(&self) -> isize {
        self.size_in_bits as isize - self.bit_pos as isize
    }

    #[inline]
    pub fn bits_read(&self) -> usize {
        self.bit_pos
    }

    #[inline]
    pub fn align_to_byte(&mut self) {
        self.bit_pos = (self.bit_pos + 7) & !7;
    }

    #[inline]
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        if self.bit_pos + n > self.size_in_bits {
            return Err(DecoderError::InvalidData("bitstream overflow".into()));
        }
        self.bit_pos += n;
        Ok(())
    }

    #[inline]
    pub fn get_bits1(&mut self) -> Result<u32> {
        self.get_bits(1)
    }

    /// Read up to 32 bits.
    #[inline]
    pub fn get_bits(&mut self, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(DecoderError::InvalidData("get_bits > 32".into()));
        }
        if self.bit_pos + n > self.size_in_bits {
            return Err(DecoderError::InvalidData("bitstream overflow".into()));
        }

        let mut out: u32 = 0;
        let mut remaining = n;
        while remaining > 0 {
            let byte_idx = self.bit_pos >> 3;
            let bit_in_byte = self.bit_pos & 7; // 0..7, MSB-first
            let avail = 8 - bit_in_byte;
            let take = remaining.min(avail);

            let byte = self.buf[byte_idx] as u32;
            let shift = (avail - take) as u32;
            let mask = (1u32 << take) - 1;
            let bits = (byte >> shift) & mask;

            out = (out << take) | bits;
            self.bit_pos += take;
            remaining -= take;
        }

        Ok(out)
    }

    #[inline]
    pub fn show_bits(&self, n: usize) -> Result<u32> {
        let mut tmp = self.clone();
        tmp.get_bits(n)
    }

    #[inline]
    pub fn get_bits_long(&mut self, n: usize) -> Result<u32> {
        self.get_bits(n)
    }

    #[inline]
    pub fn get_sbits(&mut self, n: usize) -> Result<i32> {
        if n == 0 || n > 32 {
            return Err(DecoderError::InvalidData("invalid signed bit width".into()));
        }
        let raw = self.get_bits(n)?;
        let shift = 32 - n;
        Ok(((raw << shift) as i32) >> shift)
    }
}
