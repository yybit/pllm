use std::{
    fs::File,
    io::{self, BufReader, Write},
    time::Instant,
};

use pllm::{Config, GgufFile, Tokenizer, Weights, LLM};

fn main() {
    let f = File::open("testdata/gemma2b").unwrap();
    // let mmap = unsafe { Mmap::map(&f).unwrap() };
    // let reader = io::Cursor::new(&mmap[..]);
    let reader = BufReader::new(f);
    let mut gf = GgufFile::from_reader(reader).unwrap();

    let config = Config::from_gguf(&gf).unwrap();
    println!("{:?}", config.clone());

    let tokenizer = Tokenizer::from_gguf(&gf).unwrap();

    let mut weights = Weights::new(config.clone());
    weights.load_from_gguf(&mut gf, config.clone()).unwrap();

    let iterator = LLM::new(config, tokenizer, weights)
        .inference("why the sky is blue?".to_string(), 0.8)
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
