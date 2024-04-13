[![crates.io](https://img.shields.io/crates/v/pllm.svg)](https://crates.io/crates/pllm)
[![docs.rs](https://docs.rs/pllm/badge.svg)](https://docs.rs/pllm)

# Portable LLM

A rust library for LLM inference，which ported from [llama2.c](https://github.com/karpathy/llama2.c.git). For learning purposes only, it is currently not available for production. 

## Feature

* Transformer (Currently support llama2 & gemma)
* [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file format
* SIMD (Support x86_64 avx2, derived from [candle](https://github.com/huggingface/candle))
* MMAP :construction: :construction:

## Example

```
# Download testdata
make testdata
# Run example in release mode
RUSTFLAGS='-C target-cpu=native -C target-feature=+avx2' cargo run --example llama2c --release
```

output:
>a dog. She lived in a cozy hole with her family. She liked to play outside and explore new things.
One day, she saw something unusual in the sky. It was a big, shiny aeroplane. Lionce had never seen anything like it before. The aeroplane was strange and bright.
Lionion's family saw her looking at the aeroplane and asked her what she was doing there. Lionwn proudly said, "I found this unusual aeroplane. It's so pretty and shiny!"
Lionion's family smiled and told her it was the most special thing they had ever seen. She took the aeroplane and placed it in the ground.
The next day, when Liona went outside, she saw something amazing. The aeroplane had changed! It was now a big, bright orange butterfly!
Lionna was so happy. She watched the butterfly fly away and smiled. With her original adventure, she decided to take her butterfly home with her.
token/s: 204.37956204379563

## Performance

model: `tinystories15M`, prompt: `a dog`

|version|speed|os|arch|cpu|comment|
|--|--|--|--|--|--|
|0.3.0|81 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|single thread|
|0.3.1|140 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|multiple thread|
|0.4.0|204 token/s|osx|x86|2.2 GHz Quad-Core Intel Core i7|multiple thread, simd|

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