use std::{
    iter::Sum,
    simd::{f32x32, f32x8, num::SimdFloat, StdFloat},
    time::Instant,
};

use rand::Rng;
use rayon::prelude::*;

use crate::errors::RlmError;

pub trait FloatVec {
    fn rope_rotate_neox(&mut self, pos: u32, header_size: u32, kv_dim: u32)
        -> Result<(), RlmError>;
    fn get_chunk(&self, chunk_size: u32, chunk_index: u32) -> &[f32];

    fn get_mut_chunk(&mut self, chunk_size: u32, chunk_index: u32) -> &mut [f32];

    fn arg_max(&self) -> u32;

    fn sample(&self) -> u32;

    fn rms_norm(&mut self, src: &[f32], weights: &[f32]);

    // fn mat_mul(&mut self, src: &[f32], weights: &[f32]);

    fn soft_max(&mut self);

    fn accum(&mut self, other: &[f32]);

    fn scale(&mut self, rhs: f32);

    fn rope_rotate(
        &mut self,
        other: &mut [f32],
        pos: u32,
        header_size: u32,
        kv_dim: u32,
    ) -> Result<(), RlmError>;
}

impl FloatVec for [f32] {
    fn get_chunk(&self, chunk_size: u32, chunk_index: u32) -> &[f32] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &self[range]
    }

    fn get_mut_chunk(&mut self, chunk_size: u32, chunk_index: u32) -> &mut [f32] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &mut self[range]
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
        let r = rng.gen_range(0.0..1.0);
        let mut cdf = 0.0;
        for (i, &p) in self.iter().enumerate() {
            cdf = cdf + p;
            if r < cdf {
                return i as u32;
            }
        }

        self.len() as u32 - 1
    }
    fn rms_norm(&mut self, src: &[f32], weights: &[f32]) {
        let mut ss: f32 = src.iter().map(|&i| i * i).sum();

        ss /= src.len() as f32;
        ss += 1e-5;
        ss = 1.0 / ss.sqrt();

        src.iter()
            .enumerate()
            .for_each(|(i, &x)| self[i] = weights[i] * (ss * x));
    }

    // fn mat_mul(&mut self, src: &[f32], weights: &[f32]) {
    //     let n = src.len();
    //     let d = weights.len() / n;
    //     assert_eq!(self.len(), d);

    //     // let before = Instant::now();
    //     self.par_iter_mut().enumerate().for_each(|(i, value)| {
    //         *value = f32_dot_product(src, &weights[i * n..(i + 1) * n]);
    //     });
    //     // println!("Mat mul time: n={}, d={}, {:.2?}", n, d, before.elapsed());
    // }

    fn soft_max(&mut self) {
        let len = self.len();
        if len == 1 {
            self[0] = 1.0;
            return;
        }

        // find max value (for numerical stability)
        let mut max = 0.0;
        for n in self.iter() {
            max = n.max(max);
        }

        // e^x
        self.iter_mut().for_each(|v| *v = (*v - max).exp());

        // normalize
        let sum: f32 = self.iter().map(|&n| n).sum();
        self.iter_mut().for_each(|v| *v = *v / sum);
    }

    fn accum(&mut self, other: &[f32]) {
        self.iter_mut()
            .zip(other)
            .for_each(|(v1, v2)| *v1 = *v1 + *v2)
    }

    fn scale(&mut self, rhs: f32) {
        self.iter_mut().for_each(|v| *v = *v * rhs)
    }

    fn rope_rotate(
        &mut self,
        other: &mut [f32],
        pos: u32,
        header_size: u32,
        kv_dim: u32,
    ) -> Result<(), RlmError> {
        for i in (0..self.len() as usize).step_by(2) {
            let head_dim = i as u32 % header_size;
            let freq = 1.0 / 10000.0_f32.powf(head_dim as f32 / header_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
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

    // ref. https://github.com/crabml/crabml/blob/1e622975d56c7d15dc1c627ee3cb884de0dce953/crabml-core/src/backends/cpu/primitives/rope.rs#L34
    fn rope_rotate_neox(
        &mut self,
        pos: u32,
        header_size: u32,
        kv_dim: u32,
    ) -> Result<(), RlmError> {
        let head_dim = header_size as usize;
        let rope_dim = kv_dim as usize;
        self.chunks_exact_mut(head_dim).for_each(|chunk| {
            for i in 0..rope_dim / 2 {
                let freq_exponents = 2.0 * i as f32 / head_dim as f32;
                let timescale = 10000_f32.powf(freq_exponents);
                let theta = pos as f32 / timescale;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let qp0 = chunk[i];
                let qp1 = chunk[i + head_dim / 2];
                chunk[i] = qp0 * cos_theta - qp1 * sin_theta;
                chunk[i + head_dim / 2] = qp0 * sin_theta + qp1 * cos_theta;
            }
        });

        Ok(())
    }
}

// #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
// fn f32_dot_product(a: &[f32], b: &[f32]) -> f32 {
//     let mut sum = 0.0;
//     for i in (0..a.len()).step_by(8) {
//         let s1 = f32x8::from_slice(&a[i..i + 8]);
//         let s2 = f32x8::from_slice(&b[i..i + 8]);
//         sum += (s1 * s2).reduce_sum();
//     }
//     sum
// }

// #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
// fn f32_dot_product(a: &[f32], b: &[f32]) -> f32 {
//     assert_eq!(a.len(), b.len());
//     let mut product = 0.0;
//     for i in 0..a.len() {
//         product += a[i] * b[i];
//     }
//     product
// }
