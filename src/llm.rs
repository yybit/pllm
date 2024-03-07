use std::cmp;

use crate::{
    errors::RlmError, transformer::Transformer, util::FloatVecExt, Config, Tokenizer, Weights,
};

const DEFAULT_STEPS: u32 = 256;

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

    pub fn inference(
        self,
        prompt: String,
        temperature: f32,
    ) -> Result<InferenceIterator, RlmError> {
        let steps = cmp::min(self.config.seq_len, DEFAULT_STEPS);

        let prompt_tokens = self.tokenizer.bpe_encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(RlmError::Other("empty prompt".to_string()));
        }

        let iterator = InferenceIterator::new(self, prompt_tokens, steps, temperature);

        Ok(iterator)
    }
}

pub struct InferenceIterator {
    llm: LLM,
    prompt_tokens: Vec<u32>,
    steps: u32,
    temperature: f32,

    next_token: u32,
    pos: u32,
}

impl InferenceIterator {
    pub fn new(llm: LLM, prompt_tokens: Vec<u32>, steps: u32, temperature: f32) -> Self {
        let next_token = prompt_tokens[0];
        Self {
            llm,
            prompt_tokens,
            steps,
            temperature,
            next_token,
            pos: 0,
        }
    }
}

impl Iterator for InferenceIterator {
    type Item = Result<String, RlmError>;

    fn next(&mut self) -> Option<Self::Item> {
        if (self.pos != 0 && self.next_token == 1) || self.pos >= self.steps {
            return None;
        }

        let logits = match self
            .llm
            .transformer
            .run(self.next_token, self.pos, &self.llm.weights)
        {
            Ok(l) => l,
            Err(e) => {
                return Some(Err(e));
            }
        };

        let next_token = if self.pos as usize + 1 < self.prompt_tokens.len() {
            self.prompt_tokens[self.pos as usize + 1]
        } else {
            if self.temperature == 0.0 {
                logits.arg_max()
            } else {
                logits.iter_mut().for_each(|x| *x = *x / self.temperature);
                logits.soft_max();
                logits.sample()
            }
        };

        let token_str = match self.llm.tokenizer.get_token(next_token as usize) {
            Some(t) => t,
            None => {
                return Some(Err(RlmError::Other(format!(
                    "token not found, idx={}",
                    self.next_token
                ))))
            }
        };

        self.pos += 1;
        self.next_token = next_token;

        if next_token == 1 {
            None
        } else {
            Some(Ok(token_str))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{self, BufReader, Write},
        time::Instant,
    };

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
        println!("{}", tokenizer.max_token_length);

        let iterator = LLM::new(config, tokenizer, weights)
            .inference("a dog".to_string(), 0.8)
            .unwrap();
        let mut token_count = 0;
        let start = Instant::now();
        for (i, t) in iterator.enumerate() {
            print!("{}", t.unwrap());
            io::stdout().flush().unwrap();
            token_count += 1;
        }
        println!(
            "\ntoken/s: {}\n",
            (token_count as u64 - 1) / start.elapsed().as_secs()
        );
    }
}
