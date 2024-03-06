use rand::Rng;

use crate::errors::RlmError;

pub trait FloatVecExt<F> {
    fn rms_norm(&mut self, src: &[F], weights: &[F]);
    fn get_chunk(&self, chunk_size: i32, chunk_index: i32) -> &[f32];
    fn get_mut_chunk(&mut self, chunk_size: i32, chunk_index: i32) -> &mut [f32];
    fn mat_mul(&mut self, src: &[F], weights: &[F]);
    fn soft_max(&mut self);
    fn accum(&mut self, other: &[F]);
    fn arg_max(&self) -> usize;
    fn sample(&self) -> usize;
    fn rope_rotate(
        &mut self,
        other: &mut [F],
        pos: i32,
        header_size: i32,
        kv_dim: i32,
    ) -> Result<(), RlmError>;
}

impl FloatVecExt<f32> for [f32] {
    fn accum(&mut self, other: &[f32]) {
        self.iter_mut()
            .zip(other)
            .for_each(|(v1, v2)| *v1 = *v1 + v2)
    }

    fn get_chunk(&self, chunk_size: i32, chunk_index: i32) -> &[f32] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &self[range]
    }

    fn get_mut_chunk(&mut self, chunk_size: i32, chunk_index: i32) -> &mut [f32] {
        let range = (chunk_size * chunk_index) as usize..(chunk_size * (chunk_index + 1)) as usize;
        &mut self[range]
    }

    fn rms_norm(&mut self, src: &[f32], weights: &[f32]) {
        let mut ss: f32 = src.iter().map(|&i| i * i).sum();

        ss /= src.len() as f32;
        ss += 1e-5;
        ss = 1.0 / ss.sqrt();

        src.iter()
            .enumerate()
            .for_each(|(i, x)| self[i] = weights[i] * (ss * x));
    }

    fn mat_mul(&mut self, src: &[f32], weights: &[f32]) {
        let n = src.len();
        let d = weights.len() / n;
        for i in 0..d {
            let mut sum = 0.0;
            for j in 0..n {
                sum += weights[i * n + j] * src[j];
            }
            self[i] = sum;
        }
    }

    fn soft_max(&mut self) {
        let len = self.len();
        if len == 1 {
            self[0] = 1.0;
            return;
        }
        // find max value (for numerical stability)
        let max = match self.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(v) => v,
            None => return,
        }
        .clone();

        // e^x
        self.iter_mut().for_each(|v| *v = (*v - max).exp());

        // normalize
        let sum: f32 = self.iter().sum();
        self.iter_mut().for_each(|v| *v = *v / sum);
    }

    fn arg_max(&self) -> usize {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn sample(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen_range(0.0..1.0);
        let mut cdf = 0.0;
        for (i, p) in self.iter().enumerate() {
            cdf += p;
            if r < cdf {
                return i;
            }
        }

        self.len() - 1
    }

    fn rope_rotate(
        &mut self,
        other: &mut [f32],
        pos: i32,
        header_size: i32,
        kv_dim: i32,
    ) -> Result<(), RlmError> {
        if self.len() != other.len() {
            return Err(RlmError::Other(format!(
                "ROPE rotate warning: self len: {}, other len: {}",
                self.len(),
                other.len()
            )));
        }

        for i in (0..self.len() as usize).step_by(2) {
            let head_dim = i as i32 % header_size;
            let freq = 1.0 / 10000.0_f32.powf(head_dim as f32 / header_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if (i as i32) < kv_dim { 2 } else { 1 };
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
