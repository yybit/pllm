use crate::{errors::RlmError, util::FloatVecExt, Config, Weights};

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
        wo: &[f32],
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        rms_att_weight: &[f32],
        rms_ffn_weight: &[f32],
        pos: u32,
    ) -> Result<(), RlmError> {
        let k = self.k.get_mut(pos);
        let v = self.v.get_mut(pos);

        // attention rmsnorm
        self.xb.rms_norm(x, rms_att_weight);

        // qkv matmuls for this position
        self.q.mat_mul(self.xb.as_slice(), wq);
        k.mat_mul(self.xb.as_slice(), wk);
        v.mat_mul(self.xb.as_slice(), wv);

        // apply RoPE rotation to the q and k vectors for each head
        self.q.rope_rotate(k, pos, self.header_size, self.kv_dim)?;

        // multihead attention. iterate over all heads
        for (h, header) in self.heads.iter_mut().enumerate() {
            let xb = self.xb.get_mut_chunk(self.header_size, h as u32);
            xb.iter_mut().for_each(|item| *item = 0.0);

            let q = self.q.get_chunk(self.header_size, h as u32);
            header.calculate_activation(xb, q, &self.k, &self.v, pos, h as u32, self.header_size);
        }

        // final matmul to get the output of the attention
        self.xb2.mat_mul(self.xb.as_slice(), wo);
        // residual connection back into x
        x.accum(self.xb2.as_slice());

        // ffn rmsnorm
        self.xb.rms_norm(x, rms_ffn_weight);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        self.hb.mat_mul(self.xb.as_slice(), w1);
        self.hb2.mat_mul(self.xb.as_slice(), w3);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        // elementwise multiply with w3(x)
        for i in 0..self.hb.len() {
            self.hb[i] = self.hb[i] * (1.0 / (1.0 + (-self.hb[i]).exp())) * self.hb2[i];
        }

        // final matmul to get the output of the ffn
        self.xb.mat_mul(self.hb.as_slice(), w2);

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

        Self {
            config,
            layers,
            x,
            logits,
        }
    }

    pub fn run(&mut self, token: u32, pos: u32, w: &Weights) -> Result<&mut [f32], RlmError> {
        let c = &self.config;

        self.x
            .copy_from_slice(&w.token_embedding_table.get_chunk(c.dim, token));

        for (lu, layer) in self.layers.iter_mut().enumerate() {
            let l = lu as u32;
            layer.forward(
                &mut self.x,
                w.wo.get_chunk(c.dim * c.dim, l),
                w.w1.get_chunk(c.dim * c.hidden_dim, l),
                w.w2.get_chunk(c.dim * c.hidden_dim, l),
                w.w3.get_chunk(c.dim * c.hidden_dim, l),
                w.wq.get_chunk(c.dim * c.dim, l),
                w.wk.get_chunk(c.dim * c.kv_dim(), l),
                w.wv.get_chunk(c.dim * c.kv_dim(), l),
                w.rms_att_weight.get_chunk(c.dim, l),
                w.rms_ffn_weight.get_chunk(c.dim, l),
                pos,
            )?;
        }

        // final rmsnorm
        let x_clone = self.x.clone();
        self.x
            .rms_norm(x_clone.as_slice(), w.rms_final_weight.as_ref());
        // classifier into logits
        self.logits
            .mat_mul(self.x.as_slice(), w.token_embedding_table.as_ref());

        Ok(&mut self.logits)
    }
}
