use byteorder::{LittleEndian, ReadBytesExt};
use std::{
    cmp,
    io::Read,
    ops::Mul,
    simd::{f32x32, f32x8, num::SimdFloat, StdFloat},
    time::Instant,
};

use half::f16;
use rayon::prelude::*;

use crate::quant::*;
use crate::{avx::vec_dot_q8_0_q8_0, errors::PllmError};

#[derive(Debug, Clone)]
pub struct TensorQ4_0(Vec<BlockQ4_0>);

impl TensorQ4_0 {
    pub fn from_reader(&mut self, mut reader: impl Read) -> Result<(), PllmError> {
        let mut deltas_buf = [0; 2];
        let mut quants_buf = [0u8; Q4_0_GROUP_SIZE / 2];

        for i in 0..self.0.len() {
            reader.read_exact(&mut deltas_buf)?;
            reader.read_exact(&mut quants_buf)?;
            self.0[i].d = f16::from_le_bytes(deltas_buf);
            self.0[i].qs.copy_from_slice(&quants_buf);
        }

        Ok(())
    }

    pub fn new(size: usize) -> Self {
        assert_eq!(size % Q4_0_GROUP_SIZE, 0);

        Self(vec![Default::default(); size / Q4_0_GROUP_SIZE])
    }

    fn len(&self) -> usize {
        self.0.len() * Q4_0_GROUP_SIZE
    }

    fn get_group(&self, i: usize) -> (f32, &[u8]) {
        let scale = self.0[i].d;
        let sub_quants = &self.0[i].qs;

        (scale.to_f32(), sub_quants)
    }

    fn get_slice(&self, offset: usize, len: usize) -> &[BlockQ4_0] {
        &self.0[offset..offset + len]
    }

    fn dot_product_q4_0(&self, other: &TensorQ4_0) -> f32 {
        assert_eq!(self.len(), other.len());

        let mut sum = 0.0;
        self.0.iter().enumerate().for_each(|(i, b)| {
            let (other_scale, other_sub_quants) = other.get_group(i);
            let scale = b.d.to_f32();

            let mut group_sum: i32 = 0;
            for (j, &n) in b.qs.iter().enumerate() {
                let other_n = other_sub_quants[j];

                let x0 = (n & 0x0F) as i32 - 8;
                let x1 = (n >> 4) as i32 - 8;

                let other_x0 = (other_n & 0x0F) as i32 - 8;
                let other_x1 = (other_n >> 4) as i32 - 8;

                group_sum += x0 * other_x0 + x1 * other_x1;
            }

            sum += group_sum as f32 * scale * other_scale;
        });
        sum
    }

    fn quantize(&mut self, nums: &[f32]) {
        assert_eq!(nums.len(), self.len());

        nums.chunks(Q4_0_GROUP_SIZE)
            .enumerate()
            .for_each(|(i, sub_nums)| {
                let max_abs_origin = sub_nums
                    .into_iter()
                    .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
                    .unwrap_or(&0.0);

                let scale = max_abs_origin / -8.0;
                let iscale = if scale != 0f32 { 1.0 / scale } else { 0.0 };

                self.0[i].d = f16::from_f32(scale);

                let half_group_size = Q4_0_GROUP_SIZE / 2;
                for j in 0..half_group_size {
                    let x0 = sub_nums[j] * iscale;
                    let x1 = sub_nums[half_group_size + j] * iscale;

                    let xi0 = cmp::min(15, (x0 + 8.5) as u8);
                    let xi1 = cmp::min(15, (x1 + 8.5) as u8);

                    self.0[i].qs[j] = xi0 | (xi1 << 4);
                }
            });
    }

    fn dequantize(&self, nums: &mut [f32], offset: usize) {
        assert!(nums.len() + offset <= self.len());
        assert_eq!(nums.len() % Q4_0_GROUP_SIZE, 0);

        let half_group_size = Q4_0_GROUP_SIZE / 2;

        for (i, b) in self.0[offset / Q4_0_GROUP_SIZE..(offset + nums.len()) / Q4_0_GROUP_SIZE]
            .iter()
            .enumerate()
        {
            let scale = b.d.to_f32();

            for (j, &n) in b.qs.iter().enumerate() {
                let x0 = (n & 0x0F) as i16 - 8;
                let x1 = (n >> 4) as i16 - 8;
                nums[i * Q4_0_GROUP_SIZE + j] = (x0 as f32) * scale;
                nums[i * Q4_0_GROUP_SIZE + half_group_size + j] = (x1 as f32) * scale;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorQ8_0(Vec<BlockQ8_0>);

impl TensorQ8_0 {
    pub fn from_reader(&mut self, mut reader: impl Read) -> Result<(), PllmError> {
        let mut deltas_buf = [0; 2];
        let mut quants_buf = [0i8; Q8_0_GROUP_SIZE];

        for i in 0..self.0.len() {
            reader.read_exact(&mut deltas_buf)?;
            reader.read_i8_into(&mut quants_buf)?;

            self.0[i].d = f16::from_le_bytes(deltas_buf);
            self.0[i].qs.copy_from_slice(&quants_buf);
        }

        Ok(())
    }

    pub fn new(size: usize) -> Self {
        assert_eq!(size % Q8_0_GROUP_SIZE, 0);

        Self(vec![Default::default(); size / Q8_0_GROUP_SIZE])
    }

    fn len(&self) -> usize {
        self.0.len() * Q8_0_GROUP_SIZE
    }

    fn get_group(&self, i: usize) -> (f32, &[i8]) {
        let scale = self.0[i].d.to_f32();
        let sub_quants = &self.0[i].qs;

        (scale, sub_quants)
    }

    fn get_slice(&self, offset: usize, len: usize) -> &[BlockQ8_0] {
        &self.0[offset..offset + len]
    }

    fn quantize(&mut self, nums: &[f32]) {
        assert_eq!(nums.len(), self.len());

        let i8_max = 127.0_f32;

        nums.chunks(Q8_0_GROUP_SIZE)
            .enumerate()
            .for_each(|(i, sub_nums)| {
                let mut max_abs = 0.0;
                for n in sub_nums {
                    max_abs = n.abs().max(max_abs);
                }

                let scale = max_abs / i8_max;
                let iscale = if scale != 0f32 { 1.0 / scale } else { 0.0 };

                self.0[i].d = f16::from_f32(scale);
                for (j, n) in sub_nums.iter().enumerate() {
                    self.0[i].qs[j] = (n * iscale).round() as i8;
                }
            });
    }

    fn dequantize(&self, nums: &mut [f32], offset: usize) {
        assert!(nums.len() + offset <= self.len());
        assert_eq!(nums.len() % Q8_0_GROUP_SIZE, 0);

        let scale_offset = offset / Q8_0_GROUP_SIZE;
        for (i, b) in self.0[offset / Q8_0_GROUP_SIZE..(offset + nums.len()) / Q8_0_GROUP_SIZE]
            .iter()
            .enumerate()
        {
            let scale = b.d.to_f32();

            for (j, &n) in b.qs.iter().enumerate() {
                nums[i * Q8_0_GROUP_SIZE + j] = n as f32 * scale;
            }
        }
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    fn dot_product_q4_0(&self, other: &TensorQ4_0, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());
        assert!(other_offset % Q4_0_GROUP_SIZE == 0);

        let other_group_offset = other_offset / Q4_0_GROUP_SIZE;

        let mut sum = 0.0;
        let half_group_size = Q4_0_GROUP_SIZE / 2;

        self.0.iter().enumerate().for_each(|(i, b)| {
            let (other_scale, other_sub_quants) = other.get_group(other_group_offset + i);
            let scale = b.d.to_f32();

            let mut group_sum: i32 = 0;

            for (j, &other_n) in other_sub_quants.iter().enumerate() {
                let x0 = b.qs[j] as i32;
                let x1 = b.qs[half_group_size + j] as i32;

                let other_x0 = (other_n & 0x0F) as i32 - 8;
                let other_x1 = (other_n >> 4) as i32 - 8;

                group_sum += x0 * other_x0 + x1 * other_x1;
            }

            sum += group_sum as f32 * scale * other_scale;
        });
        sum
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn dot_product_q4_0(&self, other: &TensorQ4_0, other_offset: usize) -> f32 {
        use crate::avx::vec_dot_q4_0_q8_0;

        assert!(self.len() + other_offset <= other.len());
        assert!(other_offset % Q4_0_GROUP_SIZE == 0);

        let other_group_offset = other_offset / Q4_0_GROUP_SIZE;

        vec_dot_q4_0_q8_0(other.get_slice(other_group_offset, self.0.len()), &self.0)

        // let mut sum = 0.0;
        // let half_group_size = Q4_0_GROUP_SIZE / 2;

        // let mut sum_buf = [0.0; 8];
        // let mut scale_buf = [0.0; 8];
        // let mut other_scale_buf = [0.0; 8];

        // self.0.iter().enumerate().for_each(|(i, b)| {
        //     let (other_scale, other_sub_quants) = other.get_group(other_group_offset + i);
        //     let scale = b.d.to_f32();

        //     let mut group_sum: i32 = 0;

        //     for (j, &other_n) in other_sub_quants.iter().enumerate() {
        //         let x0 = b.qs[j] as i32;
        //         let x1 = b.qs[half_group_size + j] as i32;

        //         let other_x0 = (other_n & 0x0F) as i32 - 8;
        //         let other_x1 = (other_n >> 4) as i32 - 8;

        //         group_sum += x0 * other_x0 + x1 * other_x1;
        //     }

        //     let buf_i = i % 8;

        //     sum_buf[buf_i] = group_sum as f32;
        //     scale_buf[buf_i] = scale;
        //     other_scale_buf[buf_i] = other_scale;

        //     if buf_i == 7 {
        //         let mul = f32x8::from_array(sum_buf)
        //             * f32x8::from_array(scale_buf)
        //             * f32x8::from_array(other_scale_buf);
        //         sum += mul.reduce_sum();
        //     }
        // });
        // sum
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn dot_product_q8_0(&self, other: &TensorQ8_0, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());
        assert!(other_offset % Q8_0_GROUP_SIZE == 0);

        let other_group_offset = other_offset / Q8_0_GROUP_SIZE;
        vec_dot_q8_0_q8_0(other.get_slice(other_group_offset, self.0.len()), &self.0)

        // let mut sum = 0.0;
        // let mut sum_buf = [0.0; 8];
        // let mut scale_buf = [0.0; 8];
        // let mut other_scale_buf = [0.0; 8];

        // self.0.iter().enumerate().for_each(|(i, b)| {
        //     let (other_scale, other_sub_quants) = other.get_group(other_group_offset + i);
        //     let scale = b.d.to_f32();

        //     let mut group_sum: i32 = 0;
        //     for (j, &n) in b.qs.iter().enumerate() {
        //         let x0 = n as i32;
        //         let other_x0 = other_sub_quants[j] as i32;

        //         group_sum += x0 * other_x0;
        //     }

        //     let buf_i = i % 8;

        //     sum_buf[buf_i] = group_sum as f32;
        //     scale_buf[buf_i] = scale;
        //     other_scale_buf[buf_i] = other_scale;

        //     if buf_i == 7 {
        //         let mul = f32x8::from_array(sum_buf)
        //             * f32x8::from_array(scale_buf)
        //             * f32x8::from_array(other_scale_buf);
        //         sum += mul.reduce_sum();
        //     }
        // });
        // sum
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    fn dot_product_q8_0(&self, other: &TensorQ8_0, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());
        assert!(other_offset % Q8_0_GROUP_SIZE == 0);

        let other_group_offset = other_offset / Q8_0_GROUP_SIZE;
        let mut sum = 0.0;
        self.0.iter().enumerate().for_each(|(i, b)| {
            let (other_scale, other_sub_quants) = other.get_group(other_group_offset + i);
            let scale = b.d.to_f32();

            let mut group_sum: i32 = 0;
            for (j, &n) in b.qs.iter().enumerate() {
                let x0 = n as i32;
                let other_x0 = other_sub_quants[j] as i32;

                group_sum += x0 * other_x0;
            }

            sum += group_sum as f32 * scale * other_scale;
        });
        sum
    }
}

#[derive(Debug, Clone)]
pub struct TensorF32(Vec<f32>);

impl TensorF32 {
    pub fn from_reader(&mut self, mut reader: impl Read) -> Result<(), PllmError> {
        reader.read_f32_into::<LittleEndian>(&mut self.0)?;

        Ok(())
    }

    pub fn new(size: usize) -> Self {
        TensorF32(vec![0.0; size])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_slice(nums: &[f32]) -> TensorF32 {
        TensorF32(nums.into())
    }

    // pub fn rms_norm(&mut self, src: TensorF32, weights: TensorF32) {
    //     let mut ss: f32 = src.0.iter().map(|&i| i * i).sum();

    //     ss /= src.0.len() as f32;
    //     ss += 1e-5;
    //     ss = 1.0 / ss.sqrt();

    //     src.0
    //         .iter()
    //         .enumerate()
    //         .for_each(|(i, &x)| self.0[i] = weights.0[i] * (ss * x));
    // }

    pub fn quantize(&mut self, nums: &[f32]) {
        assert_eq!(nums.len(), self.0.len());
        self.0.copy_from_slice(nums);
    }

    pub fn dequantize(&self, nums: &mut [f32], offset: usize) {
        assert!(nums.len() + offset <= self.0.len());

        nums.copy_from_slice(&self.0[offset..offset + nums.len()]);
    }

    #[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"))))]
    fn dot_product_f32(&self, other: &TensorF32, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());

        let mut sum = 0.0;
        for i in 0..self.len() {
            sum += self.0[i] * other.0[other_offset + i];
        }
        sum
    }

    fn dot_product_q4_0(&self, other: &TensorQ4_0, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());
        assert!(other_offset % Q4_0_GROUP_SIZE == 0);

        let other_group_offset = other_offset / Q4_0_GROUP_SIZE;

        let mut sum = 0.0;
        let half_group_size = Q4_0_GROUP_SIZE / 2;

        let mut sum_buf = [0.0; 8];
        let mut other_scale_buf = [0.0; 8];

        self.0
            .chunks(Q4_0_GROUP_SIZE)
            .enumerate()
            .for_each(|(i, sub_quants)| {
                let (other_scale, other_sub_quants) = other.get_group(other_group_offset + i);

                let mut group_sum: f32 = 0.0;

                let mut s1_buf = [0.0; 8];
                let mut s2_buf = [0.0; 8];
                for (j, &other_n) in other_sub_quants.iter().enumerate() {
                    let buf_j = j % 4;
                    let double_buf_j = buf_j << 1;
                    s1_buf[double_buf_j] = sub_quants[j];
                    s1_buf[double_buf_j + 1] = sub_quants[half_group_size + j];

                    s2_buf[double_buf_j] = ((other_n & 0x0F) as i32 - 8) as f32;
                    s2_buf[double_buf_j + 1] = ((other_n >> 4) as i32 - 8) as f32;

                    if buf_j == 3 {
                        let mul = f32x8::from_array(s1_buf) * f32x8::from_array(s2_buf);
                        group_sum += mul.reduce_sum();
                    }
                }

                let buf_i = i % 8;

                sum_buf[buf_i] = group_sum as f32;
                other_scale_buf[buf_i] = other_scale;

                if buf_i == 7 {
                    let mul = f32x8::from_array(sum_buf) * f32x8::from_array(other_scale_buf);
                    sum += mul.reduce_sum();
                }
            });
        sum
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn dot_product_f32(&self, other: &TensorF32, other_offset: usize) -> f32 {
        assert!(self.len() + other_offset <= other.len());

        self.0
            .array_chunks::<8>()
            .map(|&a| f32x8::from_array(a))
            .zip(
                other.0[other_offset..other_offset + self.len()]
                    .array_chunks::<8>()
                    .map(|&b| f32x8::from_array(b)),
            )
            .fold(f32x8::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
            .reduce_sum()
    }
}

#[derive(Clone)]
pub enum Tensor {
    F32(TensorF32),
    Q8_0(TensorQ8_0),
    Q4_0(TensorQ4_0),
    None,
}

impl Tensor {
    pub fn dot_product(&self, other: &Tensor, other_offset: usize) -> f32 {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => a.dot_product_f32(&b, other_offset),
            (Tensor::F32(a), Tensor::Q4_0(b)) => a.dot_product_q4_0(&b, other_offset),
            (Tensor::Q8_0(a), Tensor::Q8_0(b)) => a.dot_product_q8_0(&b, other_offset),
            (Tensor::Q8_0(a), Tensor::Q4_0(b)) => a.dot_product_q4_0(&b, other_offset),
            (_, _) => unimplemented!(),
        }
    }

    pub fn quantize(&mut self, nums: &[f32]) {
        match self {
            Tensor::Q8_0(v) => v.quantize(nums),
            Tensor::Q4_0(v) => v.quantize(nums),
            Tensor::F32(v) => v.quantize(nums),
            _ => unimplemented!(),
        }
    }

    pub fn dequantize(&self, nums: &mut [f32], offset: usize) {
        match self {
            Tensor::F32(v) => v.dequantize(nums, offset),
            Tensor::Q8_0(v) => v.dequantize(nums, offset),
            Tensor::Q4_0(v) => v.dequantize(nums, offset),
            Tensor::None => {}
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Tensor::F32(v) => v.0.len(),
            Tensor::Q8_0(v) => v.len(),
            Tensor::Q4_0(v) => v.len(),
            _ => 0,
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Tensor::None => true,
            _ => false,
        }
    }
}

pub trait F32TensorExt {
    fn tensor_mul(&mut self, src: &Tensor, weights: &Tensor);
    fn to_tensor(&self) -> Tensor;
}

impl F32TensorExt for [f32] {
    fn tensor_mul(&mut self, src: &Tensor, weights: &Tensor) {
        let n = src.len();
        let d = weights.len() / n;
        assert_eq!(self.len(), d);

        // let before = Instant::now();
        self.par_iter_mut().enumerate().for_each(|(i, value)| {
            *value = src.dot_product(weights, i * n);
        });
        // println!(
        //     "Tensor mul time: n={}, d={}, {:.2?}",
        //     n,
        //     d,
        //     before.elapsed()
        // );
    }

    fn to_tensor(&self) -> Tensor {
        Tensor::F32(TensorF32::from_slice(self))
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    use super::TensorQ8_0;
    use std::time::Instant;

    use super::F32TensorExt;
    use super::TensorQ4_0;

    #[test]
    fn test_vec_q8_0() {
        let nums: Vec<f32> = (-10..54).into_iter().map(|x| x as f32).collect();
        let mut vq = TensorQ8_0::new(64);
        vq.quantize(&nums);
        println!("{:?}", vq);

        let mut q_nums = vec![0f32; 64];
        vq.dequantize(&mut q_nums, 0);
        println!("{:?}", q_nums);
    }
    #[test]
    fn test_vec_q4_0() {
        let nums: Vec<f32> = (-10..54).into_iter().map(|x| x as f32).collect();
        let mut vq = TensorQ4_0::new(64);
        vq.quantize(&nums);
        println!("{:?}", vq);

        let mut q_nums = vec![0f32; 64];
        vq.dequantize(&mut q_nums, 0);
        println!("{:?}", q_nums);
    }

    #[test]
    fn test_dot_product() {
        let nums: Vec<f32> = (0..2048).into_iter().map(|x| x as f32).collect();
        let mut vq4 = TensorQ4_0::new(2048);
        vq4.quantize(&nums);

        let mut vq8 = TensorQ8_0::new(2048);
        vq8.quantize(&nums);

        let before = Instant::now();
        // let sum = vq8.dot_product_q4_0(&vq4);
        println!("Elapsed time: {:.2?}", before.elapsed());
        // println!("sum: {}", sum);
    }

    #[test]
    fn test_tensor_mul() {
        let n = 32;
        let nums: Vec<f32> = (0..n * n).into_iter().map(|x| x as f32).collect();
        let mut vq4 = TensorQ4_0::new(n * n);
        vq4.quantize(&nums);

        let nums2: Vec<f32> = (0..n).into_iter().map(|x| x as f32).collect();
        let mut vq8 = TensorQ8_0::new(n);
        vq8.quantize(&nums2);

        let mut v = vec![0.0; n];
        v.tensor_mul(&Tensor::Q8_0(vq8), &Tensor::Q4_0(vq4));
    }
}
