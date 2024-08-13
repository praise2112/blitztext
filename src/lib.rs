//!
//! blitztext
//!
//! High-performance keyword extraction and replacement in strings.
//!
//! BlitzText is a Rust library for efficient keyword processing, based on the [FlashText](https://github.com/vi3k6i5/flashtext) and  [Aho-Corasick](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm) algorithm. It's designed for high-speed operations on large volumes of text.
//!
//! ## Overview
//!
//! The main component of BlitzText is the `KeywordProcessor`, which manages a trie-based data structure for storing and matching keywords. It supports various operations including keyword addition, extraction, and replacement, with options for case sensitivity, fuzzy matching, and parallel processing.
//!
//! ## Structs
//!
//! ### [`KeywordProcessor`]
//!
//! The core struct for keyword operations.
//!
//! #### Methods
//!
//! *  `new` - Creates a new [`KeywordProcessor`] with default settings.
//! * `with_options` - Creates a new  [`KeywordProcessor`] with specified options.
//! * `add_keyword` - Adds a keyword to the processor.
//! * `remove_keyword` - Removes a keyword from the processor.
//! * `extract_keywords` - Extracts keywords from the given text.
//! * `replace_keywords` - Replaces keywords in the given text.
//! * `parallel_extract_keywords_from_texts` - Extracts keywords from multiple texts in parallel.
//!
//! ###  [`KeywordMatch`]
//!
//! Represents a matched keyword in the text.
//!
//! #### Fields
//!
//! * `keyword: &str` - The matched keyword.
//! * `similarity: f32` - The similarity score of the match (1.0 for exact matches).
//! * `start: usize` - The start index of the match in the text.
//! * `end: usize` - The end index of the match in the text.
//!
//! ## Examples
//!
//! ### Basic Usage
//!
//! ```rust
//! use blitztext::KeywordProcessor;
//!
//! fn main() {
//!     let mut processor = KeywordProcessor::new();
//!     processor.add_keyword("rust", Some("Rust Lang"));
//!     processor.add_keyword("programming", Some("Coding"));
//!
//!     let text = "I love rust programming";
//!     let matches = processor.extract_keywords(text, None);
//!
//!     for m in matches {
//!         println!("Found '{}' at [{}, {}]", m.keyword, m.start, m.end);
//!     }
//!
//!     let replaced = processor.replace_keywords(text, None);
//!     println!("Replaced text: {}", replaced);
//!     // Output: "I love Rust Lang Coding"
//! }
//! ```
//!
//! ### Fuzzy Matching
//!
//! ```rust
//! # use blitztext::KeywordProcessor;
//! # let mut processor = KeywordProcessor::new();
//! # let text = "I love rust programming";
//! let matches = processor.extract_keywords(text, Some(0.8));
//! ```
//!
//! ### Parallel Processing
//!
//! ```rust
//! # use blitztext::KeywordProcessor;
//! # let mut processor = KeywordProcessor::new();
//! let texts = vec!["Text 1", "Text 2", "Text 3"];
//! let results = processor.parallel_extract_keywords_from_texts(&texts, None);
//! ```
//!
//! ### Custom Non-Word Boundaries
//!
//! ```rust
//! use blitztext::KeywordProcessor;
//! let mut processor = KeywordProcessor::new();
//! processor.add_keyword("rust", None);
//! processor.add_keyword("programming", Some("coding"));
//!
//! let text = "I-love-rust-programming-and-1coding2";
//!
//! // Default behavior: '-' is a word separator
//! let matches = processor.extract_keywords(text, None);
//! assert_eq!(matches.len(), 2);
//!
//! // Add '-' as a non-word boundary
//! processor.add_non_word_boundary('-');
//!
//! // Now '-' is considered part of words
//! let matches = processor.extract_keywords(text, None);
//! assert_eq!(matches.len(), 0);
//! ```
//!
//! ## Performance
//!
//! BlitzText is optimized for high-speed operations, making it suitable for processing large volumes of text. It uses a trie-based data structure for efficient matching and supports parallel processing for handling multiple texts simultaneously.
//! ```
//!
//!

mod char_set;
mod keyword_processor;
mod trie;
mod utils;

pub use char_set::CharSet;
pub use keyword_processor::KeywordMatch;
pub use keyword_processor::KeywordProcessor;

use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass(name = "KeywordProcessor", module = "blitztext")]
#[derive(Serialize, Deserialize)]
struct PyKeywordProcessor(KeywordProcessor);

#[pyclass(module = "blitztext")]
#[derive(Serialize, Deserialize, Clone)]
struct PyKeywordMatch {
    #[pyo3(get, set)]
    pub keyword: String,
    #[pyo3(get, set)]
    pub similarity: f32,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

#[pymethods]
impl PyKeywordMatch {
    #[new]
    fn new(keyword: String, similarity: f32, start: usize, end: usize) -> Self {
        PyKeywordMatch {
            keyword,
            similarity,
            start,
            end,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "KeywordMatch(keyword='{}', similarity={:.2}, start={}, end={})",
            self.keyword, self.similarity, self.start, self.end
        )
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }

    fn __getnewargs__(&self) -> PyResult<(String, f32, usize, usize)> {
        Ok((self.keyword.clone(), self.similarity, self.start, self.end))
    }
}

impl From<KeywordMatch<'_>> for PyKeywordMatch {
    fn from(km: KeywordMatch) -> Self {
        PyKeywordMatch {
            keyword: km.keyword.to_string(),
            similarity: km.similarity,
            start: km.start,
            end: km.end,
        }
    }
}

#[pymethods]
impl PyKeywordProcessor {
    #[new]
    #[pyo3(signature = (case_sensitive=false, allow_overlaps=false))]
    fn new(case_sensitive: Option<bool>, allow_overlaps: Option<bool>) -> Self {
        let case_sensitive = case_sensitive.unwrap_or(false);
        let allow_overlaps = allow_overlaps.unwrap_or(false);
        PyKeywordProcessor(KeywordProcessor::with_options(
            case_sensitive,
            allow_overlaps,
        ))
    }

    #[pyo3(signature = (keyword, clean_name=None))]
    fn add_keyword(&mut self, keyword: &str, clean_name: Option<&str>) {
        self.0.add_keyword(keyword, clean_name);
    }

    fn remove_keyword(&mut self, keyword: &str) -> bool {
        self.0.remove_keyword(keyword)
    }

    #[pyo3(signature = (text, threshold=None))]
    fn extract_keywords(&self, text: &str, threshold: Option<f32>) -> Vec<PyKeywordMatch> {
        self.0
            .extract_keywords(text, threshold)
            .into_iter()
            .map(PyKeywordMatch::from)
            .collect()
    }

    #[pyo3(signature = (texts, threshold=None))]
    fn parallel_extract_keywords_from_texts(
        &self,
        texts: Vec<String>,
        threshold: Option<f32>,
    ) -> Vec<Vec<PyKeywordMatch>> {
        self.0
            .parallel_extract_keywords_from_texts(&texts, threshold)
            .into_iter()
            .map(|matches| matches.into_iter().map(PyKeywordMatch::from).collect())
            .collect()
    }

    #[pyo3(signature = (text, threshold=None))]
    fn replace_keywords(&self, text: &str, threshold: Option<f32>) -> String {
        self.0.replace_keywords(text, threshold)
    }

    #[pyo3(signature = (texts, threshold=None))]
    fn parallel_replace_keywords_from_texts(
        &self,
        texts: Vec<String>,
        threshold: Option<f32>,
    ) -> Vec<String> {
        self.0
            .parallel_replace_keywords_from_texts(&texts, threshold)
    }

    fn get_all_keywords(&self) -> HashMap<String, String> {
        self.0.get_all_keywords()
    }

    fn set_non_word_boundaries(&mut self, boundaries: Vec<char>) {
        self.0.set_non_word_boundaries(&boundaries);
    }

    fn add_non_word_boundary(&mut self, boundary: char) {
        self.0.add_non_word_boundary(boundary);
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!("KeywordProcessor(num_keywords={})", self.0.len())
    }

    // https://users.rust-lang.org/t/rust-extension-module-breaks-multiprocessing-no-pickle-support/112402
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
}

#[pymodule]
fn blitztext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeywordProcessor>()?;
    m.add_class::<PyKeywordMatch>()?;
    Ok(())
}
