use half::f16;

pub const Q4_0_GROUP_SIZE: usize = 32;
pub const Q8_0_GROUP_SIZE: usize = 32;

#[derive(Debug, Clone, Default)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; Q4_0_GROUP_SIZE / 2],
}

#[derive(Debug, Clone, Default)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; Q8_0_GROUP_SIZE],
}
