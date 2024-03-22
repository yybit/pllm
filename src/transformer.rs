use crate::{
    errors::RlmError,
    tensor::{F32TensorExt, Tensor},
    util::FloatVec,
    Config, Weights,
};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone)]
pub struct LayerCache {
    data: Vec<f32>,
    header_size: u32,
    kv_dim: u32,
    kv_mul: u32,
}

impl LayerCache {
    pub fn new(header_size: u32, seq_len: u32, kv_dim: u32, kv_mul: u32) -> Self {
        Self {
            data: vec![0.0; (kv_dim * seq_len) as usize],
            header_size,
            kv_dim,
            kv_mul,
        }
    }

    pub fn get(&self, position: u32, header_idx: u32) -> &[f32] {
        let start =
            (position * self.kv_dim + (header_idx / self.kv_mul) * self.header_size) as usize;
        &self.data[start..(start + self.header_size as usize)]
    }

    pub fn get_mut(&mut self, position: u32) -> &mut [f32] {
        self.data.get_mut_chunk(self.kv_dim, position)
    }
}

#[derive(Clone)]
pub struct Head {
    scores: Vec<f32>,
}

impl Head {
    pub fn new(seq_len: u32) -> Self {
        Self {
            scores: vec![0.0; seq_len as usize],
        }
    }

    pub fn calculate_activation(
        &mut self,
        xb: &mut [f32],
        q: &[f32],
        k: &LayerCache,
        v: &LayerCache,
        pos: u32,
        header_idx: u32,
        header_size: u32,
    ) {
        for t in 0..=pos {
            let keys = k.get(t, header_idx);
            let mut score: f32 = (0..header_size as usize)
                .into_iter()
                .map(|i| q[i] * keys[i])
                .sum();
            score = score / (header_size as f32).sqrt();
            // save the score to the attention buffer
            self.scores[t as usize] = score;
        }

        self.scores[..(pos as usize + 1)].soft_max();

        // weighted sum of the values, store back into xb
        for t in 0..=pos {
            let values = v.get(t, header_idx);
            for i in 0..header_size as usize {
                xb[i] += values[i] * self.scores[t as usize];
            }
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    /// activations (dim,)
    xb: Vec<f32>,
    /// an additional buffer just for convenience (dim,)
    xb2: Vec<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<f32>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,
    /// query (dim,)
    pub q: Vec<f32>,
    /// key (dim,)
    k: LayerCache,
    /// value (dim,)
    v: LayerCache,

    heads: Vec<Head>,
    header_size: u32,
    kv_dim: u32,
}
impl Layer {
    pub fn new(c: &Config) -> Self {
        let xb = vec![0_f32; c.dim as usize];
        let xb2 = xb.clone();

        let hb = vec![0_f32; c.hidden_dim as usize];
        let hb2 = hb.clone();

        let q = xb.clone();

        let k = LayerCache::new(c.header_size(), c.seq_len, c.kv_dim(), c.kv_mul());
        let v = LayerCache::new(c.header_size(), c.seq_len, c.kv_dim(), c.kv_mul());

        let heads = vec![Head::new(c.seq_len); c.n_heads as usize];

        Self {
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            heads,
            header_size: c.header_size(),
            kv_dim: c.kv_dim(),
        }
    }
    pub fn forward(
        &mut self,
        x: &mut [f32],
        wo: &Tensor,
        w1: &Tensor,
        w2: &Tensor,
        w3: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        rms_att_weight: &[f32],
        rms_ffn_weight: &[f32],
        pos: u32,
        xb_q: &mut Tensor,
        hb_q: &mut Tensor,
        is_gemma: bool,
    ) -> Result<(), RlmError> {
        let k = self.k.get_mut(pos);
        let v = self.v.get_mut(pos);

        // attention rmsnorm
        self.xb.rms_norm(x, rms_att_weight);

        if xb_q.is_none() {
            let xq = self.xb.to_tensor();
            self.q.tensor_mul(&xq, wq);
            k.tensor_mul(&xq, wk);
            v.tensor_mul(&xq, wv);
        } else {
            xb_q.quantize(&self.xb);
            self.q.tensor_mul(xb_q, wq);
            k.tensor_mul(xb_q, wk);
            v.tensor_mul(xb_q, wv);
        }

        // apply RoPE rotation to the q and k vectors for each head
        if is_gemma {
            self.q
                .rope_rotate_neox(pos, self.header_size, self.kv_dim)?;
            k.rope_rotate_neox(pos, self.header_size, self.kv_dim)?;
        } else {
            self.q.rope_rotate(k, pos, self.header_size, self.kv_dim)?;
        }

        // multihead attention. iterate over all heads
        self.xb
            .par_chunks_mut(self.header_size as usize)
            .enumerate()
            .zip(self.heads.par_iter_mut())
            .for_each(|((h, xb_chunk), header)| {
                xb_chunk.iter_mut().for_each(|item| *item = 0.0);

                let q = self.q.get_chunk(self.header_size, h as u32);
                header.calculate_activation(
                    xb_chunk,
                    q,
                    &self.k,
                    &self.v,
                    pos,
                    h as u32,
                    self.header_size,
                );
            });

        // final matmul to get the output of the attention
        if xb_q.is_none() {
            self.xb2.tensor_mul(&self.xb.to_tensor(), wo);
        } else {
            xb_q.quantize(&self.xb);
            self.xb2.tensor_mul(xb_q, wo);
        }
        // residual connection back into x
        x.accum(self.xb2.as_slice());

        // ffn rmsnorm
        self.xb.rms_norm(x, rms_ffn_weight);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        if xb_q.is_none() {
            self.hb.tensor_mul(&self.xb.to_tensor(), w1);
            self.hb2.tensor_mul(&self.xb.to_tensor(), w3);
        } else {
            xb_q.quantize(&self.xb);
            self.hb.tensor_mul(xb_q, w1);
            self.hb2.tensor_mul(xb_q, w3);
        }

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        // elementwise multiply with w3(x)
        if is_gemma {
            for i in 0..self.hb.len() {
                let item = self.hb[i];
                let tmp = 0.797_884_560_802_865_4 * item * (1.0 + 0.044715 * item * item);

                self.hb[i] = 0.5 * item * (1.0 + tmp.tanh()) * self.hb2[i];
            }
        } else {
            for i in 0..self.hb.len() {
                self.hb[i] = self.hb[i] * (1.0 / (1.0 + (-self.hb[i]).exp())) * self.hb2[i];
            }
        }

        // final matmul to get the output of the ffn
        if hb_q.is_none() {
            self.xb.tensor_mul(&self.hb.to_tensor(), w2);
        } else {
            hb_q.quantize(&self.hb);
            self.xb.tensor_mul(hb_q, w2);
        }

        x.accum(self.xb.as_slice());

        Ok(())
    }
}

pub struct Transformer {
    config: Config,
    layers: Vec<Layer>,
    /// activation at current time stamp (dim,)
    x: Vec<f32>,
    /// output logits
    logits: Vec<f32>,
}

impl Transformer {
    pub fn new(config: Config) -> Self {
        let x = vec![0_f32; config.dim as usize];
        let logits = vec![0_f32; config.vocab_size as usize];
        let layers = vec![Layer::new(&config); config.n_layers as usize];
        let layers = vec![Layer::new(&config); config.n_layers as usize];

        Self {
            config,
            layers,
            x,
            logits,
        }
    }

    pub fn run(&mut self, token: u32, pos: u32, w: &Weights) -> Result<&mut [f32], RlmError> {
        let c = &self.config;

        w.token_embedding_table
            .dequantize(&mut self.x, (c.dim * token) as usize);

        if self.config.is_gemma() {
            self.x.scale((c.dim as f32).sqrt());
        }

        let mut xb_q = w.make_quantize_tensor(c.dim as usize);
        let mut hb_q = w.make_quantize_tensor(c.hidden_dim as usize);

        for (lu, layer) in self.layers.iter_mut().enumerate() {
            let l = lu as u32;
            let before = Instant::now();
            layer.forward(
                &mut self.x,
                &w.wo[lu],
                &w.w1[lu],
                &w.w2[lu],
                &w.w3[lu],
                &w.wq[lu],
                &w.wk[lu],
                &w.wv[lu],
                w.rms_att_weight.get_chunk(c.dim, l),
                w.rms_ffn_weight.get_chunk(c.dim, l),
                pos,
                &mut xb_q,
                &mut hb_q,
                self.config.is_gemma(),
            )?;
            // println!("Layer time: l={}, {:.2?}", l, before.elapsed());
        }

        // final rmsnorm
        let x_clone = self.x.clone();
        self.x
            .rms_norm(x_clone.as_slice(), w.rms_final_weight.as_ref());
        // classifier into logits

        if xb_q.is_none() {
            self.logits
                .tensor_mul(&self.x.to_tensor(), &w.token_embedding_table);
        } else {
            xb_q.quantize(&self.x);
            self.logits.tensor_mul(&xb_q, &w.token_embedding_table);
        }

        Ok(&mut self.logits)
    }
}
