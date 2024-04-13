#![feature(portable_simd)]
#![feature(array_chunks)]

// extern crate blas_src;
use std::simd::num::SimdInt;
use std::simd::{f32x16, f32x32, f32x8, i16x16, i16x8, i32x16, i32x4, i32x8};
use std::simd::{f32x4, num::SimdFloat, StdFloat};
use std::time::Instant;

// use ndarray::linalg::Dot;
// use ndarray::{ArcArray, CowArray};

// pub fn f32_dot_prod_ndarray(a: &[f32], b: &[f32]) -> f32 {
//     let ar = CowArray::from(a);
//     let br = CowArray::from(b);
//     ar.dot(&br)
// }

// pub fn i32_dot_prod_ndarray(a: &[i32], b: &[i32]) -> i32 {
//     let ar = CowArray::from(a);
//     let br = CowArray::from(b);
//     ar.dot(&br)
// }

pub fn f32_dot_prod_test() {
    let N = 2048;
    let K = 256128;
    let a: Vec<f32> = (0..N).map(|i| ((i as f32) / 100.0).sin()).collect();
    let b: Vec<f32> = (0..N).map(|i| ((i as f32) / 100.0).cos()).collect();
    let mut result = 0.0;
    let mut before = Instant::now();
    for _ in 0..K {
        result = f32_dot_prod_simd_4(&a, &b);
    }
    println!("f32x4 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = f32_dot_prod_simd_8(&a, &b);
    }
    println!("f32x8 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = f32_dot_prod_simd_16(&a, &b);
    }
    println!("f32x16 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = f32_dot_prod_simd_32(&a, &b);
    }
    println!("f32x32 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = f32_dot_prod(&a, &b);
    }
    println!("f32 plain time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    // before = Instant::now();
    // for _ in 0..K {
    //     result = f32_dot_prod_ndarray(&a, &b);
    // }
    // println!("f32 ndarray time: {:.2?}", before.elapsed());
    // println!("result: {}", result);
}

pub fn i16_dot_prod_test() {
    let N = 2048;
    let K = 256128;
    let a: Vec<i16> = (0..N).collect();
    let b: Vec<i16> = (0..N).collect();
    let mut result: i32 = 0;
    let mut before = Instant::now();
    for _ in 0..K {
        result = i16_dot_prod_simd_8(&a, &b);
    }
    println!("i16x8 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = i16_dot_prod_simd_16(&a, &b);
    }
    println!("i16x16 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = i16_dot_prod(&a, &b);
    }
    println!("i16 plain time: {:.2?}", before.elapsed());
    println!("result: {}", result);
}

pub fn i32_dot_prod_test() {
    let N = 2048;
    let K = 256128;
    let a: Vec<i32> = (0..N).collect();
    let b: Vec<i32> = (0..N).collect();
    let mut result = 0;
    let mut before = Instant::now();
    for _ in 0..K {
        result = i32_dot_prod_simd_4(&a, &b);
    }
    println!("i32x4 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = i32_dot_prod_simd_8(&a, &b);
    }
    println!("i32x8 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = i32_dot_prod_simd_16(&a, &b);
    }
    println!("i32x16 time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    before = Instant::now();
    for _ in 0..K {
        result = i32_dot_prod(&a, &b);
    }
    println!("i32 plain time: {:.2?}", before.elapsed());
    println!("result: {}", result);

    // before = Instant::now();
    // for _ in 0..K {
    //     result = i32_dot_prod_ndarray(&a, &b);
    // }
    // println!("i32 ndarray time: {:.2?}", before.elapsed());

    // println!("result: {}", result);
}

fn f32_dot_prod_simd_4(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<4>()
        .map(|&a| f32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| f32x4::from_array(b)))
        .fold(f32x4::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

fn i32_dot_prod_simd_4(a: &[i32], b: &[i32]) -> i32 {
    a.array_chunks::<4>()
        .map(|&a| i32x4::from_array(a))
        .zip(b.array_chunks::<4>().map(|&b| i32x4::from_array(b)))
        .fold(i32x4::splat(0), |acc, (a, b)| a * b + acc)
        .reduce_sum()
}

fn f32_dot_prod_simd_8(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<8>()
        .map(|&a| f32x8::from_array(a))
        .zip(b.array_chunks::<8>().map(|&b| f32x8::from_array(b)))
        .fold(f32x8::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

fn i32_dot_prod_simd_8(a: &[i32], b: &[i32]) -> i32 {
    a.array_chunks::<8>()
        .map(|&a| i32x8::from_array(a))
        .zip(b.array_chunks::<8>().map(|&b| i32x8::from_array(b)))
        .fold(i32x8::splat(0), |acc, (a, b)| a * b + acc)
        .reduce_sum()
}

fn i16_dot_prod_simd_8(a: &[i16], b: &[i16]) -> i32 {
    a.array_chunks::<8>()
        .map(|&a| i16x8::from_array(a))
        .zip(b.array_chunks::<8>().map(|&b| i16x8::from_array(b)))
        .fold(i16x8::splat(0), |acc, (a, b)| a * b + acc)
        .reduce_sum() as i32
}

fn f32_dot_prod_simd_16(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<16>()
        .map(|&a| f32x16::from_array(a))
        .zip(b.array_chunks::<16>().map(|&b| f32x16::from_array(b)))
        .fold(f32x16::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

fn i32_dot_prod_simd_16(a: &[i32], b: &[i32]) -> i32 {
    a.array_chunks::<16>()
        .map(|&a| i32x16::from_array(a))
        .zip(b.array_chunks::<16>().map(|&b| i32x16::from_array(b)))
        .fold(i32x16::splat(0), |acc, (a, b)| a * b + acc)
        .reduce_sum()
}

fn i16_dot_prod_simd_16(a: &[i16], b: &[i16]) -> i32 {
    a.array_chunks::<16>()
        .map(|&a| i16x16::from_array(a))
        .zip(b.array_chunks::<16>().map(|&b| i16x16::from_array(b)))
        .fold(i16x16::splat(0), |acc, (a, b)| a * b + acc)
        .reduce_sum() as i32
}

fn f32_dot_prod_simd_32(a: &[f32], b: &[f32]) -> f32 {
    a.array_chunks::<32>()
        .map(|&a| f32x32::from_array(a))
        .zip(b.array_chunks::<32>().map(|&b| f32x32::from_array(b)))
        .fold(f32x32::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum()
}

fn f32_dot_prod(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

fn i32_dot_prod(a: &[i32], b: &[i32]) -> i32 {
    assert_eq!(a.len(), b.len());
    let mut product = 0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

fn i16_dot_prod(a: &[i16], b: &[i16]) -> i32 {
    assert_eq!(a.len(), b.len());
    let mut product = 0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product as i32
}

fn main() {
    f32_dot_prod_test();
    i32_dot_prod_test();
    i16_dot_prod_test();
}
