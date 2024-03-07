[![crates.io](https://img.shields.io/crates/v/pllm.svg)](https://crates.io/crates/pllm)
[![docs.rs](https://docs.rs/pllm/badge.svg)](https://docs.rs/pllm)

# Portable LLM

A rust library for LLM inference，which ported from [llama2.c](https://github.com/karpathy/llama2.c.git). Support [gguf](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is working in progress.

## Example

```
# Download testdata
make testdata
# Run example in release mode
cargo run -p example --release
```

output:
> a dog and a cat. They were friends, but one day, they had a quarrel. The dog said, "I want to play with the ball." The cat said, "No, I want to play with the ball."
The dog and the cat were both sad. They did not want to share. They both wanted the ball. They did not know what to do.
Then, a bird flew down. The bird said, "Why don't you both play with the ball? You can be friends and share." The dog and the cat thought about it. They decided to share the ball.
The dog and the cat took turns playing with the ball. They played together and were happy. They didn't have to quarrel anymore. They learned that sharing is good.
token/s: 81

## Performance

|version|speed|os|arch|cpu|
|--|--|--|--|--|
|0.3.0|81 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|

## Usage

```rust
use std::{
    fs::File,
    io::{self, BufReader, Write},
};

use pllm::{Config, Tokenizer, Weights, LLM};

// Load config from model
let f = File::open("testdata/stories15M.bin").unwrap();
let mut reader = BufReader::new(f);
let config = Config::from_reader(&mut reader).unwrap();

// Load weights from model
let mut weights = Weights::new(config.clone());
weights.load_data(&mut reader).unwrap();

// Load tokenizer
let tokenizer_file = File::open("testdata/tokenizer.bin").unwrap();
let tokenizer_reader = BufReader::new(tokenizer_file);
let tokenizer = Tokenizer::from_reader(config.vocab_size as usize, tokenizer_reader).unwrap();

// Generate text by prompt
let iterator = LLM::new(config, tokenizer, weights)
    .inference("a dog".to_string(), 0.8)
    .unwrap();
for i in iterator {
    print!("{}", i.unwrap());
    io::stdout().flush().unwrap();
}
```