use std::{
    cmp,
    io::{self, Write},
    time::Instant,
};

use crate::{
    errors::RlmError, transformer::Transformer, util::FloatVecExt, Config, Tokenizer, Weights,
};

const DEFAULT_STEPS: i32 = 256;

pub struct LLM {
    config: Config,
    tokenizer: Tokenizer,
    weights: Weights,
    transformer: Transformer,
}

impl LLM {
    pub fn new(config: Config, tokenizer: Tokenizer, weights: Weights) -> Self {
        let transformer = Transformer::new(config.clone());

        Self {
            config,
            tokenizer,
            weights,
            transformer,
        }
    }

    pub fn inference(&mut self, prompt: String, temperature: f32) -> Result<(), RlmError> {
        let steps = cmp::min(self.config.seq_len, DEFAULT_STEPS) as usize;

        let prompt_tokens = self.tokenizer.bpe_encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(RlmError::Other("empty prompt".to_string()));
        }

        let start = Instant::now();

        let mut next_token = prompt_tokens[0];
        for pos in 0..steps {
            let logits = self
                .transformer
                .run(next_token as i32, pos as i32, &self.weights)?;

            next_token = if pos + 1 < prompt_tokens.len() {
                prompt_tokens[pos + 1]
            } else {
                if temperature == 0.0 {
                    logits.arg_max()
                } else {
                    logits.iter_mut().for_each(|x| *x = *x / temperature);
                    logits.soft_max();
                    logits.sample()
                }
            };

            let token_str = match self.tokenizer.get_token(next_token) {
                Some(t) => t,
                None => {
                    return Err(RlmError::Other(format!(
                        "token not found, idx={}",
                        next_token
                    )))
                }
            };
            if next_token == 1 {
                break;
            }

            print!("{}", &token_str);
            io::stdout().flush()?;
        }

        println!(
            "\ntoken/s: {}\n",
            (steps as u64 - 1) / start.elapsed().as_secs()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use crate::llm::{Weights, LLM};

    use super::{Config, Tokenizer};

    #[test]
    fn test_reader() {
        let f = File::open("testdata/stories15M.bin").unwrap();
        let mut reader = BufReader::new(f);
        let config = Config::from_reader(&mut reader).unwrap();
        println!("{:?}", config);

        let mut weights = Weights::new(config.clone());
        weights.load_data(&mut reader).unwrap();

        let tokenizer_file = File::open("testdata/tokenizer.bin").unwrap();
        let tokenizer_reader = BufReader::new(tokenizer_file);

        let tokenizer =
            Tokenizer::from_reader(config.vocab_size as usize, tokenizer_reader).unwrap();
        println!(
            "{} {} {}",
            tokenizer.max_token_length, tokenizer.vocab_size, tokenizer._total_token_length
        );

        let mut llm = LLM::new(config, tokenizer, weights);
        llm.inference("a dog".to_string(), 0.8).unwrap();
    }
}
