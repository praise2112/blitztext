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
//!

mod char_set;
mod compiled_trie;
mod fuzzy;
mod keyword_processor;
mod trie;
mod utils;

pub use char_set::CharSet;
pub use keyword_processor::KeywordMatch;
pub use keyword_processor::KeywordProcessor;

use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyString};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass(name = "KeywordProcessor", module = "blitztext")]
#[derive(Serialize, Deserialize)]
struct PyKeywordProcessor(KeywordProcessor);

#[pyclass(module = "blitztext")]
struct PyKeywordMatch {
    keyword: Py<PyString>,
    #[pyo3(get, set)]
    pub similarity: f32,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

/// Pickled form of a `PyKeywordMatch`.
#[derive(Serialize, Deserialize)]
struct KeywordMatchState {
    keyword: String,
    similarity: f32,
    start: usize,
    end: usize,
}

#[pymethods]
impl PyKeywordMatch {
    #[new]
    fn new(py: Python<'_>, keyword: String, similarity: f32, start: usize, end: usize) -> Self {
        PyKeywordMatch {
            keyword: PyString::new(py, &keyword).unbind(),
            similarity,
            start,
            end,
        }
    }

    #[getter]
    fn keyword(&self, py: Python<'_>) -> Py<PyString> {
        self.keyword.clone_ref(py)
    }

    #[setter]
    fn set_keyword(&mut self, py: Python<'_>, keyword: String) {
        self.keyword = PyString::new(py, &keyword).unbind();
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "KeywordMatch(keyword='{}', similarity={:.2}, start={}, end={})",
            self.keyword.bind(py).to_cow()?,
            self.similarity,
            self.start,
            self.end
        ))
    }

    pub fn __setstate__(&mut self, py: Python<'_>, state: Bound<'_, PyBytes>) -> PyResult<()> {
        let state: KeywordMatchState = deserialize(state.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        *self = PyKeywordMatch::new(py, state.keyword, state.similarity, state.start, state.end);
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let state = KeywordMatchState {
            keyword: self.keyword.bind(py).to_cow()?.into_owned(),
            similarity: self.similarity,
            start: self.start,
            end: self.end,
        };
        Ok(PyBytes::new(py, &serialize(&state).unwrap()))
    }

    fn __getnewargs__(&self, py: Python<'_>) -> (Py<PyString>, f32, usize, usize) {
        (self.keyword.clone_ref(py), self.similarity, self.start, self.end)
    }
}

/// Converts matches to Python objects, sharing one string per distinct keyword.
fn to_python_matches(
    py: Python<'_>,
    matches: Vec<KeywordMatch<'_>>,
    keywords: &mut FxHashMap<(*const u8, usize), Py<PyString>>,
) -> Vec<PyKeywordMatch> {
    matches
        .into_iter()
        .map(|keyword_match| {
            let keyword = keywords
                .entry((keyword_match.keyword.as_ptr(), keyword_match.keyword.len()))
                .or_insert_with(|| PyString::new(py, keyword_match.keyword).unbind());
            PyKeywordMatch {
                keyword: keyword.clone_ref(py),
                similarity: keyword_match.similarity,
                start: keyword_match.start,
                end: keyword_match.end,
            }
        })
        .collect()
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
    fn extract_keywords(
        &self,
        py: Python<'_>,
        text: &str,
        threshold: Option<f32>,
    ) -> PyResult<Vec<PyKeywordMatch>> {
        let matches = self.0.extract_keywords(text, checked(threshold)?);
        Ok(to_python_matches(py, matches, &mut FxHashMap::default()))
    }

    #[pyo3(signature = (texts, threshold=None))]
    fn parallel_extract_keywords_from_texts(
        &self,
        py: Python<'_>,
        texts: Vec<PyBackedStr>,
        threshold: Option<f32>,
    ) -> PyResult<Vec<Vec<PyKeywordMatch>>> {
        let threshold = checked(threshold)?;
        let mut keywords = FxHashMap::default();
        Ok(py
            .detach(|| self.0.parallel_extract_keywords_from_texts(&texts, threshold))
            .into_iter()
            .map(|matches| to_python_matches(py, matches, &mut keywords))
            .collect())
    }

    #[pyo3(signature = (text, threshold=None))]
    fn replace_keywords(&self, text: &str, threshold: Option<f32>) -> PyResult<String> {
        Ok(self.0.replace_keywords(text, checked(threshold)?))
    }

    #[pyo3(signature = (texts, threshold=None))]
    fn parallel_replace_keywords_from_texts(
        &self,
        py: Python<'_>,
        texts: Vec<PyBackedStr>,
        threshold: Option<f32>,
    ) -> PyResult<Vec<String>> {
        let threshold = checked(threshold)?;
        Ok(py.detach(|| self.0.parallel_replace_keywords_from_texts(&texts, threshold)))
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
        *self = deserialize(state.as_bytes()).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
}

fn checked(threshold: Option<f32>) -> PyResult<Option<f32>> {
    match threshold {
        Some(threshold) if !(0.0..=1.0).contains(&threshold) => {
            Err(PyValueError::new_err("Threshold must be between 0 and 1"))
        }
        threshold => Ok(threshold),
    }
}

#[pymodule]
fn blitztext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeywordProcessor>()?;
    m.add_class::<PyKeywordMatch>()?;
    Ok(())
}
