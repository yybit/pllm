use std::{collections::HashMap, io::Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::errors::RlmError;

#[derive(Debug)]
pub struct Tokenizer {
    pub(crate) max_token_length: u32,
    pub(crate) vocab_size: usize,
    vocab: Vec<String>,
    token_id_map: HashMap<String, usize>,
    scores: Vec<f32>,

    pub(crate) _total_token_length: u32,
}

impl Tokenizer {
    pub fn get_token(&self, index: usize) -> Option<String> {
        self.vocab.get(index).cloned()
    }

    pub fn from_reader(vocab_size: usize, mut reader: impl Read) -> Result<Self, RlmError> {
        let max_token_length = reader.read_u32::<LittleEndian>()?;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut token_id_map = HashMap::with_capacity(vocab_size);

        let mut _total_token_length: u32 = 0;
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
            _total_token_length += len as u32;
        }

        Ok(Self {
            max_token_length,
            vocab_size,
            vocab,
            token_id_map,
            scores,
            _total_token_length,
        })
    }

    pub fn bpe_encode(&self, text: String) -> Result<Vec<usize>, RlmError> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(1);
        let dummy_prefix = self
            .token_id_map
            .get(" ")
            .ok_or(RlmError::Other(
                "dummy prefix ' '  not found in vocab".to_string(),
            ))?
            .clone();
        tokens.push(dummy_prefix);

        for c in text.chars() {
            let id = self
                .token_id_map
                .get(&c.to_string())
                .ok_or(RlmError::Other(format!("{} not found in vocab", c)))?
                .clone();
            tokens.push(id);
        }

        loop {
            let mut best_score = -1e10 as f32;
            let mut best_id = 0;
            let mut best_idx = None;

            if tokens.len() > 0 {
                for i in 0..tokens.len() - 1 {
                    let merge_token =
                        format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                    if let Some(&id) = self.token_id_map.get(&merge_token) {
                        if self.scores[id] > best_score {
                            best_score = self.scores[id];
                            best_id = id;
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
