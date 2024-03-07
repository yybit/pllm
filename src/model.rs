use std::io::Read;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::{errors::RlmError, gguf};

#[derive(Debug, Clone)]
pub struct Config {
    /// transformer dimension
    pub(crate) dim: u32,
    /// for ffn layers
    pub(crate) hidden_dim: u32,
    /// number of layers
    pub(crate) n_layers: u32,
    /// number of query headers
    pub(crate) n_heads: u32,
    /// number of key/value heads (can be < query heads because of multiquery)
    pub(crate) n_kv_heads: u32,
    /// vocabulary size, usually 256 (byte-level)
    pub vocab_size: u32,
    /// max sequence length
    pub(crate) seq_len: u32,
}

impl Config {
    pub fn from_reader(mut reader: impl Read) -> Result<Self, RlmError> {
        let dim = reader.read_u32::<LittleEndian>()?;
        let hidden_dim = reader.read_u32::<LittleEndian>()?;
        let n_layers = reader.read_u32::<LittleEndian>()?;
        let n_heads = reader.read_u32::<LittleEndian>()?;
        let n_kv_heads = reader.read_u32::<LittleEndian>()?;
        let vocab_size = reader.read_u32::<LittleEndian>()?;
        let seq_len = reader.read_u32::<LittleEndian>()?;

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }

    pub fn from_gguf(gf: gguf::GgufFile) -> Result<Self, RlmError> {
        let md = gf.metadata();

        let dim = md.get_u32_result("llama.embedding_length")?;
        let hidden_dim = md.get_u32_result("llama.feed_forward_length")?;
        let n_layers = md.get_u32_result("llama.block_count")?;
        let n_heads = md.get_u32_result("llama.attention.head_count")?;
        let n_kv_heads = md.get_u32_result("llama.attention.head_count_kv")?;
        let vocab_size = md.get_u32_result("tokenizer.ggml.tokens")?;
        let seq_len = md.get_u32_result("llama.context_length")?;

        let norm_rms_eps = md.get_f32_result("llama.attention.layer_norm_rms_epsilon")?;
        let rope_dim = md.get_u32_result("llama.rope.dimension_count")?;
        println!("rms eps: {}, rope dim: {}", norm_rms_eps, rope_dim);

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }
    pub fn header_size(&self) -> u32 {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> u32 {
        (self.dim * self.n_kv_heads) / self.n_heads
    }

    /// integer multiplier of the kv sharing in multiquery
    pub fn kv_mul(&self) -> u32 {
        self.n_heads / self.n_kv_heads
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
    // pub(crate) freq_cis_real: Vec<f32>,
    // pub(crate) freq_cis_imag: Vec<f32>,
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
        // let freq_cis_real = vec![0_f32; (c.seq_len * (c.dim / c.n_heads) / 2) as usize];
        // let freq_cis_imag = freq_cis_real.clone();

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
            // freq_cis_real,
            // freq_cis_imag,
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
