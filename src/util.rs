use std::iter::Sum;

use num_traits::{cast, Float, Num};
use rand::Rng;
use rayon::prelude::*;

use crate::errors::RlmError;

pub trait NumVecExt<N> {
    fn rms_norm(&mut self, src: &[N], weights: &[N]);
    fn get_chunk(&self, chunk_size: u32, chunk_index: u32) -> &[N];
    fn get_mut_chunk(&mut self, chunk_size: u32, chunk_index: u32) -> &mut [N];
    fn mat_mul(&mut self, src: &[N], weights: &[N]);
    fn soft_max(&mut self);
    fn accum(&mut self, other: &[N]);
    fn arg_max(&self) -> u32;
    fn sample(&self) -> u32;
    fn rope_rotate(
        &mut self,
        other: &mut [N],
        pos: u32,
        header_size: u32,
        kv_dim: u32,
    ) -> Result<(), RlmError>;
}

impl<N> NumVecExt<N> for [N]
where
    N: Num + Sum + Copy + PartialOrd + Float + Send + Sync,
{
    fn rms_norm(&mut self, src: &[N], weights: &[N]) {
        let mut ss: N = src.iter().map(|&i| i * i).sum();

        ss = ss / cast(src.len()).unwrap();
        ss = ss + cast(1e-5).unwrap();
        ss = cast::<f32, N>(1.0).unwrap() / ss.sqrt();

        src.iter()
            .enumerate()
            .for_each(|(i, &x)| self[i] = weights[i] * (ss * x));
    }

    fn get_chunk(&self, chunk_size: u32, chunk_index: u32) -> &[N] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &self[range]
    }

    fn get_mut_chunk(&mut self, chunk_size: u32, chunk_index: u32) -> &mut [N] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &mut self[range]
    }

    fn mat_mul(&mut self, src: &[N], weights: &[N]) {
        let n = src.len();
        let d = weights.len() / n;
        assert_eq!(self.len(), d);

        self.par_iter_mut().enumerate().for_each(|(i, value)| {
            let mut sum: N = cast(0.0).unwrap();
            for j in 0..n {
                sum = sum + weights[i * n + j] * src[j];
            }
            *value = sum;
        });
        // for i in 0..d {
        //     let mut sum: N = cast(0.0).unwrap();
        //     for j in 0..n {
        //         sum = sum + weights[i * n + j] * src[j];
        //     }
        //     self[i] = sum;
        // }
    }

    fn soft_max(&mut self) {
        let len = self.len();
        if len == 1 {
            self[0] = cast(1.0).unwrap();
            return;
        }
        // find max value (for numerical stability)
        let max = match self.iter().max_by(|&a, &b| a.partial_cmp(b).unwrap()) {
            Some(v) => v,
            None => return,
        }
        .clone();

        // e^x
        self.iter_mut().for_each(|v| *v = (*v - max).exp());

        // normalize
        let sum: N = self.iter().map(|&n| n).sum();
        self.iter_mut().for_each(|v| *v = *v / sum);
    }

    fn accum(&mut self, other: &[N]) {
        self.iter_mut()
            .zip(other)
            .for_each(|(v1, v2)| *v1 = *v1 + *v2)
    }

    fn arg_max(&self) -> u32 {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap_or(0)
    }

    fn sample(&self) -> u32 {
        let mut rng = rand::thread_rng();
        let r = cast::<f32, N>(rng.gen_range(0.0..1.0)).unwrap();
        let mut cdf: N = cast(0.0).unwrap();
        for (i, &p) in self.iter().enumerate() {
            cdf = cdf + p;
            if r < cdf {
                return i as u32;
            }
        }

        self.len() as u32 - 1
    }

    fn rope_rotate(
        &mut self,
        other: &mut [N],
        pos: u32,
        header_size: u32,
        kv_dim: u32,
    ) -> Result<(), RlmError> {
        if self.len() != other.len() {
            return Err(RlmError::Other(format!(
                "ROPE rotate warning: self len: {}, other len: {}",
                self.len(),
                other.len()
            )));
        }

        for i in (0..self.len() as usize).step_by(2) {
            let head_dim = i as u32 % header_size;
            let freq = 1.0 / 10000.0_f32.powf(head_dim as f32 / header_size as f32);
            let val = pos as f32 * freq;
            let fcr = cast(val.cos()).unwrap();
            let fci = cast(val.sin()).unwrap();
            let rotn = if (i as u32) < kv_dim { 2 } else { 1 };
            for v in 0..rotn {
                let dst = if v == 0 {
                    self.as_mut()
                } else {
                    other.as_mut()
                };
                let v0 = dst[i];
                let v1 = dst[i + 1];
                dst[i] = v0 * fcr - v1 * fci;
                dst[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        Ok(())
    }
}
