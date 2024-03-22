[![crates.io](https://img.shields.io/crates/v/pllm.svg)](https://crates.io/crates/pllm)
[![docs.rs](https://docs.rs/pllm/badge.svg)](https://docs.rs/pllm)

# Portable LLM

A rust library for LLM inference，which ported from [llama2.c](https://github.com/karpathy/llama2.c.git). For learning purposes only, it is currently not available for production. 

## Feature

* Transformer (Currently support llama2 & gemma)
* [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file format
* SIMD (Only support x86_64 avx2, derived from [candle](https://github.com/huggingface/candle))
* MMAP ::construction::::construction::

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

llama 
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

gemma
```rust
    let f = File::open("testdata/gemma2b").unwrap();
    // let mmap = unsafe { Mmap::map(&f).unwrap() };
    // let reader = io::Cursor::new(&mmap[..]);
    let reader = BufReader::new(f);
    let mut gf = GgufFile::from_reader(reader).unwrap();

    let config = Config::from_gguf(&gf).unwrap();
    // println!("{:?}", config.clone());

    let tokenizer = Tokenizer::from_gguf(&gf).unwrap();

    let mut weights = Weights::new(config.clone());
    weights.load_from_gguf(&mut gf, config.clone()).unwrap();

    let args: Vec<String> = env::args().collect();
    let iterator = LLM::new(config, tokenizer, weights)
        .inference(
            args.get(1)
                .unwrap_or(&"why the sky is blue?".to_string())
                .to_string(),
            0.8,
        )
        .unwrap();

    let mut token_count = 0;
    let start = Instant::now();
    for (_, t) in iterator.enumerate() {
        print!("{}", t.unwrap());
        io::stdout().flush().unwrap();
        token_count += 1;
    }
    println!(
        "\ntoken/s: {}\n",
        (token_count as f64 - 1.0) / start.elapsed().as_millis() as f64 * 1000.0
```

## Development

```
# build
RUSTFLAGS='-C target-cpu=native -C target-feature=+avx2' cargo build --release
# cross build
RUSTFLAGS='-C target-cpu=native -C target-feature=+avx2' cargo zigbuild --release --target x86_64-unknown-linux-musl
```