[![crates.io](https://img.shields.io/crates/v/pllm.svg)](https://crates.io/crates/pllm)
[![docs.rs](https://docs.rs/pllm/badge.svg)](https://docs.rs/pllm)

# Portable LLM

A rust library for LLM inference，which ported from [llama2.c](https://github.com/karpathy/llama2.c.git). Reading model from [gguf](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is working in progress.

## Example

```
# Download testdata
make testdata
# Run example in release mode
cd example
cargo run --release
```

output:
> a dog named Mark. Mark loved to play with his ball. One day, Mark saw a big cat. The cat was very grumpy. Mark wanted to play with the cat.
Mark said, "Hi, cat! Do you want to play?" The grumpy cat said, "No, I don't want to play. Go away." Mark was sad and walked away.
Mark saw a ball and started to play with it. He kicked the ball and it went far away. The grumpy cat saw the ball and started to play with it too. They both had fun playing together.
token/s: 140.46822742474916

## Performance

model: `tinystories15M`, prompt: `a dog`

|version|speed|os|arch|cpu|comment|
|--|--|--|--|--|--|
|0.3.0|81 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|single thread|
|0.3.1|140 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|multiple thread|

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

// Generate text from prompts
let iterator = LLM::new(config, tokenizer, weights)
    .inference("a dog".to_string(), 0.8)
    .unwrap();
for i in iterator {
    print!("{}", i.unwrap());
    io::stdout().flush().unwrap();
}
```