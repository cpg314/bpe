mod counter;
use counter::Counter;
pub mod bpe;

use std::path::Path;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

type Word = String;
type Token = String;
pub type TokenID = usize;

const UNK: TokenID = usize::MAX;
const UNK_STR: &str = "UNK";

pub trait Tokenizer: Sized {
    fn tokens_as_string<T: IntoIterator<Item = TokenID>>(
        &self,
        tokens: T,
    ) -> impl Iterator<Item = String>;
    fn tokenize(&self, line: &str) -> impl Iterator<Item = TokenID>;
    fn train(lines: impl IntoIterator<Item = String>, vocab_size: usize) -> Self;
    fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()>;
    fn load(path: impl AsRef<Path>) -> anyhow::Result<Self>;
}
