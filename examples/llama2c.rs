use std::{
    fs::File,
    io::{self, BufReader, Write},
    time::Instant,
};

use pllm::{Config, Tokenizer, Weights, LLM};

fn main() {
    let f = File::open("testdata/stories15M.bin").unwrap();
    let mut reader = BufReader::new(f);
    let config = Config::from_reader(&mut reader).unwrap();
    println!("{:?}", config);

    let mut weights = Weights::new(config.clone());
    weights.load_data(&mut reader).unwrap();

    let tokenizer_file = File::open("testdata/tokenizer.bin").unwrap();
    let tokenizer_reader = BufReader::new(tokenizer_file);

    let tokenizer = Tokenizer::from_reader(config.vocab_size as usize, tokenizer_reader).unwrap();

    let iterator = LLM::new(config, tokenizer, weights)
        .inference("a dog".to_string(), 0.8)
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
    );
}
