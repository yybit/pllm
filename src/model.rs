use std::io::{Read, Seek};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::gguf::GgmlType;
use crate::tensor::{Tensor, TensorF32, TensorQ8_0};
use crate::util::FloatVec;
use crate::{errors::PllmError, gguf};

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

    pub(crate) norm_rms_eps: f32,
    pub(crate) rope_dim: u32,

    arch: String,
}

impl Config {
    pub fn is_gemma(&self) -> bool {
        self.arch.eq("gemma")
    }

    pub fn from_reader(mut reader: impl Read) -> Result<Self, PllmError> {
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
            norm_rms_eps: 1e-5,
            rope_dim: dim / n_heads,
            arch: "".to_string(),
        })
    }

    pub fn from_gguf<R: Read + Seek>(gf: &gguf::GgufFile<R>) -> Result<Self, PllmError> {
        let md = gf.metadata();

        let arch = md.get_string_result("general.architecture")?;

        let dim = md.get_u32_result(&format!("{}.embedding_length", arch))?;
        let hidden_dim = md.get_u32_result(&format!("{}.feed_forward_length", arch))?;
        let n_layers = md.get_u32_result(&format!("{}.block_count", arch))?;
        let n_heads = md.get_u32_result(&format!("{}.attention.head_count", arch))?;
        let n_kv_heads = md.get_u32_result(&format!("{}.attention.head_count_kv", arch))?;
        let vocab_size = md.get_string_array_result("tokenizer.ggml.tokens")?.len() as u32;
        let seq_len = md.get_u32_result(&format!("{}.context_length", arch))?;

        let norm_rms_eps =
            md.get_f32_result(&format!("{}.attention.layer_norm_rms_epsilon", arch))?;
        let rope_dim = md
            .get_u32_result(&format!("{}.rope.dimension_count", arch))
            .unwrap_or(dim / n_heads);
        // println!("rms eps: {}, rope dim: {}", norm_rms_eps, rope_dim);

        Ok(Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            norm_rms_eps,
            rope_dim,
            arch,
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
    pub(crate) token_embedding_table: Tensor,
    /// n_layers * dim
    pub(crate) rms_att_weight: Vec<f32>,
    /// n_layers * dim
    pub(crate) rms_ffn_weight: Vec<f32>,
    /// n_layers * dim * dim
    pub(crate) wq: Vec<Tensor>,
    /// n_layers * dim * dim
    pub(crate) wk: Vec<Tensor>,
    /// n_layers * dim * dim
    pub(crate) wv: Vec<Tensor>,
    /// n_layers * dim * dim
    pub(crate) wo: Vec<Tensor>,

    /// ffn gate: n_layers * dim * hidden_dim
    pub(crate) w1: Vec<Tensor>,
    /// ffn down: n_layers * hidden_dim * dim
    pub(crate) w2: Vec<Tensor>,
    /// ffn up: n_layers * dim * hidden_dim
    pub(crate) w3: Vec<Tensor>,
    /// (dim, )
    pub(crate) rms_final_weight: Vec<f32>,
    // pub(crate) freq_cis_real: Vec<f32>,
    // pub(crate) freq_cis_imag: Vec<f32>,
    config: Config,

    quantization_version: GgmlType,

    pub(crate) output_weight: Tensor,
}

impl Weights {
    pub fn new(c: Config) -> Self {
        // let token_embedding_table = vec![0_f32; (c.vocab_size * c.dim) as usize];
        let token_embedding_table = Tensor::None;

        let rms_att_weight = vec![0_f32; (c.n_layers * c.dim) as usize];
        let rms_ffn_weight = rms_att_weight.clone();

        // let wq = vec![0_f32; (c.n_layers * c.dim * c.dim) as usize];
        // let wo = wq.clone();

        // let wk = vec![0_f32; (c.n_layers * c.dim * c.kv_dim()) as usize];
        // let wv = wk.clone();

        // let w1 = vec![0_f32; (c.n_layers * c.dim * c.hidden_dim) as usize];
        // let w2 = w1.clone();
        // let w3 = w1.clone();
        let wq = vec![Tensor::None; c.n_layers as usize];
        let wo = wq.clone();
        let wk = wq.clone();
        let wv = wq.clone();
        let w1 = wq.clone();
        let w2 = wq.clone();
        let w3 = wq.clone();

        let rms_final_weight = vec![0_f32; c.dim as usize];
        // let freq_cis_real = vec![0_f32; (c.seq_len * (c.dim / c.n_heads) / 2) as usize];
        // let freq_cis_imag = freq_cis_real.clone();

        let output_weight = Tensor::None;

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
            config: c,
            quantization_version: GgmlType::F32,
            output_weight,
        }
    }

    pub fn load_data(&mut self, mut reader: impl Read) -> Result<(), PllmError> {
        let mut t = TensorF32::new((self.config.vocab_size * self.config.dim) as usize);
        t.from_reader(&mut reader)?;
        self.token_embedding_table = Tensor::F32(t);

        reader.read_f32_into::<LittleEndian>(&mut self.rms_att_weight)?;

        let dim = self.config.dim as usize;
        let hidden_dim = self.config.hidden_dim as usize;
        let kv_dim = self.config.kv_dim() as usize;

        for x in self.wq.iter_mut() {
            let mut t = TensorF32::new(dim * dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        for x in self.wk.iter_mut() {
            let mut t = TensorF32::new(dim * kv_dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        for x in self.wv.iter_mut() {
            let mut t = TensorF32::new(dim * kv_dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        for x in self.wo.iter_mut() {
            let mut t = TensorF32::new(dim * dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        // reader.read_f32_into::<LittleEndian>(&mut self.wq)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.wk)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.wv)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.wo)?;
        reader.read_f32_into::<LittleEndian>(&mut self.rms_ffn_weight)?;

        for x in self.w1.iter_mut() {
            let mut t = TensorF32::new(dim * hidden_dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        for x in self.w2.iter_mut() {
            let mut t = TensorF32::new(dim * hidden_dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }

        for x in self.w3.iter_mut() {
            let mut t = TensorF32::new(dim * hidden_dim);
            t.from_reader(&mut reader)?;
            *x = Tensor::F32(t);
        }
        // reader.read_f32_into::<LittleEndian>(&mut self.w1)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.w2)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.w3)?;
        reader.read_f32_into::<LittleEndian>(&mut self.rms_final_weight)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.freq_cis_real)?;
        // reader.read_f32_into::<LittleEndian>(&mut self.freq_cis_imag)?;

        Ok(())
    }

    pub fn load_from_gguf<R: Read + Seek>(
        &mut self,
        gf: &mut gguf::GgufFile<R>,
        c: Config,
    ) -> Result<(), PllmError> {
        let qv = gf
            .metadata()
            .get_u32_result("general.quantization_version")?;
        self.quantization_version = GgmlType::try_from(qv)?;

        self.token_embedding_table = gf.get_tensor("token_embd.weight")?;

        for i in 0..c.n_layers as usize {
            self.wq[i] = gf.get_tensor(&format!("blk.{}.attn_q.weight", i))?;
            self.wk[i] = gf.get_tensor(&format!("blk.{}.attn_k.weight", i))?;
            self.wv[i] = gf.get_tensor(&format!("blk.{}.attn_v.weight", i))?;
            self.wo[i] = gf.get_tensor(&format!("blk.{}.attn_output.weight", i))?;
            self.w1[i] = gf.get_tensor(&format!("blk.{}.ffn_gate.weight", i))?;
            self.w2[i] = gf.get_tensor(&format!("blk.{}.ffn_down.weight", i))?;
            self.w3[i] = gf.get_tensor(&format!("blk.{}.ffn_up.weight", i))?;

            gf.get_tensor(&format!("blk.{}.attn_norm.weight", i))?
                .dequantize(self.rms_att_weight.get_mut_chunk(c.dim, i as u32), 0);

            gf.get_tensor(&format!("blk.{}.ffn_norm.weight", i))?
                .dequantize(self.rms_ffn_weight.get_mut_chunk(c.dim, i as u32), 0);
        }

        gf.get_tensor("output_norm.weight")?
            .dequantize(&mut self.rms_final_weight, 0);

        self.output_weight = match gf.get_tensor("output.weight") {
            Ok(t) => t,
            Err(PllmError::TensorNotFound(_)) => Tensor::None,
            Err(e) => return Err(e),
        };

        Ok(())
    }

    pub fn make_quantize_tensor(&self, size: usize) -> Tensor {
        match self.quantization_version {
            GgmlType::Q4_0 | GgmlType::Q8_0 => Tensor::Q8_0(TensorQ8_0::new(size)),
            _ => Tensor::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Read, Seek},
    };

    use crate::{gguf::GgufFile, Config};

    fn get_gguf() -> GgufFile<BufReader<File>> {
        let f = File::open("testdata/gemma2b").unwrap();
        let reader = BufReader::new(f);
        GgufFile::from_reader(reader).unwrap()
    }

    #[test]
    fn test_config_from_gguf() {
        let gf = get_gguf();
        let config = Config::from_gguf(&gf).unwrap();
        println!("{:?}", config);
    }
}
