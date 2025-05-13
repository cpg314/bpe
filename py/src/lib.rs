use itertools::Itertools;
use pyo3::{exceptions::PyFileNotFoundError, prelude::*};
use rayon::prelude::*;
use tracing::*;

use tokenizers::{TokenID, Tokenizer};

#[pyclass(name = "Tokenizer")]
struct TokenizerPy {
    inner: tokenizers::bpe::Tokenizer,
}
#[pymethods]
// TODO: Improve error handling
impl TokenizerPy {
    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: tokenizers::bpe::Tokenizer::load(path)
                .map_err(|e| PyFileNotFoundError::new_err(e.to_string()))?,
        })
    }
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyFileNotFoundError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn train(lines: &Bound<'_, pyo3::types::PyAny>, vocab_size: usize) -> PyResult<Self> {
        let lines = lines
            .try_iter()?
            .map(|l| PyResult::Ok(l?.extract::<String>()?))
            .filter_map(|r| match r {
                Ok(r) => Some(r),
                Err(e) => {
                    warn!("Ignoring invalid argument {}", e);
                    None
                }
            });
        Ok(Self {
            inner: tokenizers::Tokenizer::train(lines, vocab_size),
        })
    }
    fn tokenize(&self, line: &str) -> Vec<TokenID> {
        self.inner.tokenize(line).collect()
    }
    fn tokenize_text(&self, text: &str) -> Vec<TokenID> {
        text.par_lines()
            .map(|line| self.tokenize(line))
            .flatten()
            .collect()
    }
    fn tokens_as_strings(&self, tokens: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
        let tokens: Vec<TokenID> = tokens
            .try_iter()?
            .map(|t| PyResult::Ok(t?.extract::<TokenID>()?))
            .try_collect()?;
        Ok(self.inner.tokens_as_string(tokens).collect())
    }
}
#[pymodule(name = "tokenizers")]
fn entrypoint(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TokenizerPy>()?;
    Ok(())
}
