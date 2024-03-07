use std::{collections::HashMap, io::Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::{errors::RlmError, gguf::GgufFile};

#[derive(Debug)]
pub struct Tokenizer {
    #[allow(dead_code)]
    pub(crate) max_token_length: u32,
    vocab: Vec<String>,
    scores: Vec<f32>,
    token_id_map: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn get_token(&self, index: usize) -> Option<String> {
        self.vocab.get(index).cloned()
    }

    pub fn from_gguf(gf: GgufFile) -> Result<Self, RlmError> {
        let md = gf.metadata();

        let vocab = md.get_string_array_result("tokenizer.ggml.tokens")?;
        let token_id_map = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect::<HashMap<_, _>>();
        let scores = md.get_f32_array_result("tokenizer.ggml.scores")?;

        let bos_token = md.get_u32_result("tokenizer.ggml.bos_token_id")?;
        let eos_token = md.get_u32_result("tokenizer.ggml.eos_token_id")?;
        println!("bos: {}, eos: {}", bos_token, eos_token);

        Ok(Self {
            max_token_length: 0,
            vocab,
            scores,
            token_id_map,
        })
    }

    pub fn from_reader(vocab_size: usize, mut reader: impl Read) -> Result<Self, RlmError> {
        let max_token_length = reader.read_u32::<LittleEndian>()?;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut token_id_map = HashMap::with_capacity(vocab_size);

        for i in 0..vocab_size {
            let score = reader.read_f32::<LittleEndian>()?;
            let len = reader.read_i32::<LittleEndian>()?;
            let mut buf = vec![0; len as usize];
            reader.read_exact(&mut buf)?;
            let value = String::from_utf8_lossy(&buf).to_string();

            if len as u32 > max_token_length {
                println!(
                    "Warning: unexpected token length greater than {}, i={}",
                    max_token_length, i,
                )
            }

            scores.push(score);
            vocab.push(value.clone());
            token_id_map.insert(value, i);
        }

        Ok(Self {
            max_token_length,
            vocab,
            scores,
            token_id_map,
        })
    }

    pub fn bpe_encode(&self, text: String) -> Result<Vec<u32>, RlmError> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(1);
        let dummy_prefix = self
            .token_id_map
            .get(" ")
            .ok_or(RlmError::Other(
                "dummy prefix ' '  not found in vocab".to_string(),
            ))?
            .clone();
        tokens.push(dummy_prefix as u32);

        for c in text.chars() {
            let id = self
                .token_id_map
                .get(&c.to_string())
                .ok_or(RlmError::Other(format!("{} not found in vocab", c)))?
                .clone();
            tokens.push(id as u32);
        }

        loop {
            let mut best_score = -1e10 as f32;
            let mut best_id = 0;
            let mut best_idx = None;

            if tokens.len() > 0 {
                for i in 0..tokens.len() - 1 {
                    let merge_token = format!(
                        "{}{}",
                        self.vocab[tokens[i] as usize],
                        self.vocab[tokens[i + 1] as usize]
                    );
                    if let Some(&id) = self.token_id_map.get(&merge_token) {
                        if self.scores[id] > best_score {
                            best_score = self.scores[id];
                            best_id = id as u32;
                            best_idx = Some(i);
                        }
                    }
                }
            }

            match best_idx {
                Some(idx) => {
                    tokens[idx] = best_id;
                    tokens.remove(idx + 1);
                }
                None => break,
            }
        }

        Ok(tokens)
    }
}
