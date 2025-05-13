use std::io::Write;
use std::path::Path;

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::{Deserialize, Serialize};
use tracing::*;

use crate::{Counter, Token, TokenID, Word, UNK, UNK_STR};

type IndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;
type BiMap<K, V> = bimap::BiHashMap<K, V, rustc_hash::FxBuildHasher>;

type Pair = (String, String);
type Splits<'a> = HashMap<&'a Word, Vec<Token>>;
type Merges = IndexMap<Pair, String>;

const APPROXIMATE_VOCAB_SIZE: bool = false;

fn most_frequent_pair<'a>(
    splits: &'a Splits,
    word_freqs: &'a Counter<Word>,
) -> Option<(&'a Token, &'a Token)> {
    let mut freqs = Counter::<(&String, &String)>::default();
    for (word, freq) in word_freqs.iter() {
        let split = &splits[word];
        if split.len() < 2 {
            continue;
        }
        for p in split.iter().tuple_windows::<(&String, &String)>() {
            freqs.increment(p, *freq);
        }
    }
    freqs.most_common().copied()
}

fn normalize_line(line: &str) -> String {
    line.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[derive(Deserialize, Serialize)]
pub struct Tokenizer {
    merges: Merges,
    pub tokens: BiMap<TokenID, Token>,
}
impl std::fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Tokenizer with {} tokens and {} merges",
            self.tokens.len(),
            self.merges.len()
        )
    }
}

impl crate::Tokenizer for Tokenizer {
    fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        Ok(std::fs::write(path, bincode::serialize(&self)?)?)
    }
    fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(&std::fs::read(path)?)?)
    }
    fn tokens_as_string<T: IntoIterator<Item = TokenID>>(
        &self,
        // TODO: Cow
        tokens: T,
    ) -> impl Iterator<Item = String> {
        tokens.into_iter().map(|t_id| {
            self.tokens
                .get_by_left(&t_id)
                .cloned()
                .unwrap_or(UNK_STR.into())
        })
    }
    fn tokenize(&self, line: &str) -> impl Iterator<Item = TokenID> {
        // TODO: Avoid allocating
        let line = normalize_line(line);
        let mut splits: Vec<Vec<Token>> = line
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .map(String::from)
                    .chain(std::iter::once(String::from(" ")))
                    .collect()
            })
            .collect();
        for (pair, merge) in &self.merges {
            for split in &mut splits {
                let mut i = 0;
                while i < split.len() - 1 {
                    if split[i] == pair.0 && split[i + 1] == pair.1 {
                        // TODO: Avoid cloning
                        split.splice(i..=i + 1, [merge.clone()]);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        splits
            .into_iter()
            .flatten()
            .map(|token| self.tokens.get_by_right(&token).copied().unwrap_or(UNK))
    }
    fn train(lines: impl IntoIterator<Item = String>, vocab_size: usize) -> Self {
        let start = std::time::Instant::now();
        info!("Training tokenizer");
        info!("Processing data");
        let mut word_freqs = Counter::<Word>::default();
        for line in lines {
            for word in normalize_line(&line).split_whitespace() {
                // TODO: without allocating?
                word_freqs.insert(word.into());
            }
        }
        let mut vocab =
            HashSet::<Token>::with_capacity_and_hasher(vocab_size, rustc_hash::FxBuildHasher);
        vocab.extend(word_freqs.keys().flat_map(|w| w.chars()).map(Token::from));
        let mut splits: Splits = word_freqs
            .keys()
            .map(|word| (word, word.chars().map(Token::from).collect()))
            .collect();
        info!("Done processing data, computing merges");

        let mut merges = Merges::with_capacity_and_hasher(vocab_size, rustc_hash::FxBuildHasher);
        let progress = indicatif::ProgressBar::no_length();
        progress.set_style(
            indicatif::ProgressStyle::with_template("[{elapsed_precise}] {msg}   {per_sec}")
                .unwrap(),
        );
        let mut stats: Option<std::io::BufWriter<std::fs::File>> = None;
        // let mut stats = std::fs::File::create("stats.csv")
        //     .inspect_err(|e| warn!("Failed to create stats file: {}", e))
        //     .ok()
        //     .map(std::io::BufWriter::new);
        while vocab.len() < vocab_size {
            progress.set_message(format!("Vocab size: {}", vocab.len()));
            progress.inc(1);
            let Some(pair) =
                most_frequent_pair(&splits, &word_freqs).map(|(a, b)| (a.clone(), b.clone()))
            else {
                break;
            };
            let joined: String = pair.0.clone() + &pair.1;
            debug!("Merging {:?}", pair);
            // Update splits
            for split in splits.values_mut() {
                if split.len() == 1 {
                    continue;
                }
                let mut i = 0;
                while i < split.len() - 1 {
                    if pair.0 == split[i] && pair.1 == split[i + 1] {
                        split.splice(i..=i + 1, [joined.clone()]);
                    } else {
                        i += 1;
                    }
                }
            }
            // Update vocab and merges
            if APPROXIMATE_VOCAB_SIZE {
                vocab.insert(joined.clone());
            }
            merges.insert(pair, joined);
            if !APPROXIMATE_VOCAB_SIZE {
                vocab = splits.values().flatten().map(Token::from).collect();
            }
            if vocab.len() % 100 == 0 {
                if let Some(stats) = &mut stats {
                    stats
                        .write_all(
                            format!(
                                "{},{},{}\n",
                                vocab.len(),
                                start.elapsed().as_secs_f32(),
                                merges.len()
                            )
                            .as_bytes(),
                        )
                        .unwrap();
                }
            }
        }
        progress.finish();

        vocab.insert(" ".into());
        let tokens: BiMap<_, _> = vocab.into_iter().sorted().enumerate().collect();
        info!(
            "Stopped after {} iterations, with {} tokens",
            progress.position(),
            tokens.len()
        );
        info!(elapsed = ?start.elapsed(), "Done training tokenizer");
        Self { merges, tokens }
    }
}
