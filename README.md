[![crates.io](https://img.shields.io/crates/v/pllm.svg)](https://crates.io/crates/pllm)
[![docs.rs](https://docs.rs/pllm/badge.svg)](https://docs.rs/pllm)

# Portable LLM

Rust rewrite of [llama2.c](https://github.com/karpathy/llama2.c.git)

## Download testdata

```
make testdata
```

## Example

```rust
let f = File::open("testdata/stories15M.bin").unwrap();
let mut reader = BufReader::new(f);
let config = Config::from_reader(&mut reader).unwrap();

let mut weights = Weights::new(config.clone());
weights.load_data(&mut reader).unwrap();

let tokenizer_file = File::open("testdata/tokenizer.bin").unwrap();
let tokenizer_reader = BufReader::new(tokenizer_file);
let tokenizer = Tokenizer::from_reader(config.vocab_size as usize, tokenizer_reader).unwrap();

let mut llm = LLM::new(config, tokenizer, weights);
llm.inference("a dog".to_string(), 0.8).unwrap();
```
