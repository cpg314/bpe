use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use rayon::prelude::*;
use tracing::*;

use tokenizers::Tokenizer;

#[derive(Parser)]
struct Flags {
    #[clap(long)]
    vocab_size: usize,
    #[clap(long, num_args=1..)]
    corpus: Vec<PathBuf>,
    #[clap(long)]
    no_cache: bool,
    #[clap(long)]
    parallel: bool,
    #[clap(long)]
    tokenize_doc: Option<PathBuf>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = Flags::parse();
    if let Err(e) = main_impl(&args) {
        error!("Failed: {}", e);
        std::process::exit(1);
    }
}

fn main_impl(args: &Flags) -> anyhow::Result<()> {
    let files = args
        .corpus
        .iter()
        .filter_map(|f| std::fs::File::open(f).ok())
        .flat_map(|f| BufReader::new(f).lines())
        .filter_map(Result::ok);

    let file = PathBuf::from(format!("tokenizer-{}.tokenizer", args.vocab_size));
    let tokenizer = if !args.no_cache && file.exists() {
        warn!(?file, "Loading tokenizer");
        tokenizers::bpe::Tokenizer::load(file)?
    } else {
        let tokenizer = tokenizers::bpe::Tokenizer::train(files, args.vocab_size);
        tokenizer.save(file)?;
        tokenizer
    };

    info!("{}", tokenizer);

    if let Some(doc) = &args.tokenize_doc {
        let start = std::time::Instant::now();
        let doc = std::fs::read_to_string(doc)?;

        let n_tokens: usize = if args.parallel {
            doc.par_lines()
                .map(|line| tokenizer.tokenize(line).count())
                .sum()
        } else {
            doc.lines()
                .map(|line| tokenizer.tokenize(line).count())
                .sum()
        };

        info!(
            "Total tokens: {}, {:.2} tokens/s",
            n_tokens,
            n_tokens as f64 / start.elapsed().as_secs_f64()
        )
    }

    Ok(())
}
