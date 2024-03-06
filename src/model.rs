use std::io::Read;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::errors::RlmError;

#[derive(Debug, Clone)]
pub struct Config {
    /// transformer dimension
    pub(crate) dim: i32,
    /// for ffn layers
    pub(crate) hidden_dim: i32,
    /// number of layers
    pub(crate) n_layers: i32,
    /// number of query headers
    pub(crate) n_headers: i32,
    /// number of key/value heads (can be < query heads because of multiquery)
    pub(crate) n_kv_headers: i32,
    /// vocabulary size, usually 256 (byte-level)
    pub vocab_size: i32,
    /// max sequence length
    pub(crate) seq_len: i32,
}

impl Config {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let dim = reader.read_i32::<LittleEndian>()?;
        let hidden_dim = reader.read_i32::<LittleEndian>()?;
        let n_layers = reader.read_i32::<LittleEndian>()?;
        let n_headers = reader.read_i32::<LittleEndian>()?;
        let n_kv_headers = reader.read_i32::<LittleEndian>()?;
        let vocab_size = reader.read_i32::<LittleEndian>()?;
        let seq_len = reader.read_i32::<LittleEndian>()?;

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_headers,
            n_kv_headers,
            vocab_size,
            seq_len,
        })
    }
    pub fn header_size(&self) -> i32 {
        self.dim / self.n_headers
    }

    pub fn kv_dim(&self) -> i32 {
        (self.dim * self.n_kv_headers) / self.n_headers
    }

    /// integer multiplier of the kv sharing in multiquery
    pub fn kv_mul(&self) -> i32 {
        self.n_headers / self.n_kv_headers
    }
}

pub struct Weights {
    /// vocab_size * dim
    pub(crate) token_embedding_table: Vec<f32>,
    pub(crate) rms_att_weight: Vec<f32>,
    pub(crate) rms_ffn_weight: Vec<f32>,
    /// n_layers * dim * dim
    pub(crate) wq: Vec<f32>,
    /// n_layers * dim * dim
    pub(crate) wk: Vec<f32>,
    /// n_layers * dim * dim
    pub(crate) wv: Vec<f32>,
    /// n_layers * dim * dim
    pub(crate) wo: Vec<f32>,
    pub(crate) w1: Vec<f32>,
    pub(crate) w2: Vec<f32>,
    pub(crate) w3: Vec<f32>,
    pub(crate) rms_final_weight: Vec<f32>,
    pub(crate) freq_cis_real: Vec<f32>,
    pub(crate) freq_cis_imag: Vec<f32>,
}

impl Weights {
    pub fn new(c: Config) -> Self {
        let token_embedding_table = vec![0_f32; (c.vocab_size * c.dim) as usize];

        let rms_att_weight = vec![0_f32; (c.n_layers * c.dim) as usize];
        let rms_ffn_weight = rms_att_weight.clone();

        let wq = vec![0_f32; (c.n_layers * c.dim * c.dim) as usize];
        let wk = wq.clone();
        let wv = wq.clone();
        let wo = wq.clone();

        let w1 = vec![0_f32; (c.n_layers * c.dim * c.hidden_dim) as usize];
        let w2 = w1.clone();
        let w3 = w1.clone();

        let rms_final_weight = vec![0_f32; c.dim as usize];
        let freq_cis_real = vec![0_f32; (c.seq_len * (c.dim / c.n_headers) / 2) as usize];
        let freq_cis_imag = freq_cis_real.clone();

        Self {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
        }
    }

    pub fn load_data(&mut self, mut reader: impl Read) -> Result<(), RlmError> {
        reader.read_f32_into::<LittleEndian>(&mut self.token_embedding_table)?;
        reader.read_f32_into::<LittleEndian>(&mut self.rms_att_weight)?;
        reader.read_f32_into::<LittleEndian>(&mut self.wq)?;
        reader.read_f32_into::<LittleEndian>(&mut self.wk)?;
        reader.read_f32_into::<LittleEndian>(&mut self.wv)?;
        reader.read_f32_into::<LittleEndian>(&mut self.wo)?;
        reader.read_f32_into::<LittleEndian>(&mut self.rms_ffn_weight)?;
        reader.read_f32_into::<LittleEndian>(&mut self.w1)?;
        reader.read_f32_into::<LittleEndian>(&mut self.w2)?;
        reader.read_f32_into::<LittleEndian>(&mut self.w3)?;
        reader.read_f32_into::<LittleEndian>(&mut self.rms_final_weight)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.freq_cis_real)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.freq_cis_imag)?;

        Ok(())
    }
}
