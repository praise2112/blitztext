use crate::char_set::CompactCharSet;
use crate::compiled_trie::CompiledTrie;
use crate::fuzzy::{self, FuzzyScratch, TrieView};
use crate::trie::TrieNode;
use crate::utils::{is_whitespace, round};
use rayon::prelude::*;
use std::collections::HashMap;
use std::mem::size_of;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// A struct for handling keyword matching, fuzzy matching, and replacement in text.
#[derive(Serialize, Deserialize)]
pub struct KeywordProcessor {
    /// Set of characters that are considered part of a word.
    non_word_boundaries: CompactCharSet,
    /// The root of the keyword trie.
    root: TrieNode,
    /// Whether the keyword matching should be case-sensitive.
    case_sensitive: bool,
    /// Whether to allow overlapping matches.
    allow_overlaps: bool,
    /// Faster read-only copy of `root`, dropped when keywords change.
    #[serde(skip)]
    compiled: OnceLock<CompiledTrie>,
    /// Number of nodes in `root`; counted on first use after deserializing.
    #[serde(skip)]
    node_count: OnceLock<usize>,
    /// Matching work done with `root` since keywords last changed, in text bytes.
    #[serde(skip)]
    work_since_change: AtomicUsize,
}

impl KeywordProcessor {
    /// Creates a new KeywordProcessor with default settings.
    ///
    /// By default, case_sensitive is set to false and allow_overlaps is set to false.
    ///
    /// # Returns
    ///
    /// A new KeywordProcessor instance with default settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let processor = KeywordProcessor::new();
    /// ```
    pub fn new() -> Self {
        Self::with_options(false, false)
    }

    /// Creates a new KeywordProcessor with specified options.
    ///
    /// # Arguments
    ///
    /// * `case_sensitive` - If true, keyword matching will be case-sensitive.
    /// * `allow_overlaps` - If true, allows overlapping matches.
    ///
    /// # Returns
    ///
    /// A new KeywordProcessor instance with the specified settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let case_sensitive_processor = KeywordProcessor::with_options(true, false);
    /// ```
    pub fn with_options(case_sensitive: bool, allow_overlaps: bool) -> Self {
        let mut non_word_boundaries = CompactCharSet::new();
        non_word_boundaries.extend('a'..='z');
        non_word_boundaries.extend('A'..='Z');
        non_word_boundaries.extend('0'..='9');
        non_word_boundaries.insert('_');
        KeywordProcessor {
            non_word_boundaries,
            root: TrieNode::new(),
            case_sensitive,
            allow_overlaps,
            compiled: OnceLock::new(),
            node_count: OnceLock::new(),
            work_since_change: AtomicUsize::new(0),
        }
    }

    /// Adds a keyword to the processor.
    ///
    /// This method inserts a keyword into the trie data structure. Here's how it works:
    ///
    /// 1. If case-insensitive, convert the keyword to lowercase.
    /// 2. Use the provided clean_name or the keyword itself if no clean_name is given.
    /// 3. Start at the root of the trie.
    /// 4. For each character in the keyword:
    ///    a. If the character doesn't exist in the current node's children, create a new node.
    ///    b. Move to the child node corresponding to this character.
    /// 5. At the last node (representing the end of the keyword), store the clean_name.
    ///
    /// This structure allows for efficient prefix matching when searching for keywords.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The keyword to add.
    /// * `clean_name` - An optional "clean" version of the keyword to return when found.
    ///                  If None, the keyword itself is used.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(m), where m is the length of the keyword.
    /// * Space complexity: O(m) in the worst case, if a completely new branch is created.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Programming Language"));
    /// ```
    pub fn add_keyword(&mut self, keyword: &str, clean_name: Option<&str>) {
        let keyword = if self.case_sensitive {
            keyword.to_string()
        } else {
            keyword.to_lowercase()
        };

        let clean_name = clean_name.unwrap_or(&keyword).to_string();

        let mut added_nodes = 0;
        let mut current_node = &mut self.root;
        // we use bytes as char instead of char because extract_keywords uses bytes
        for byte in keyword.bytes() {
            current_node = current_node.children.entry(byte as char).or_insert_with(|| {
                added_nodes += 1;
                Box::new(TrieNode::new())
            });
        }
        current_node.keyword = Some(clean_name);
        self.keywords_changed(added_nodes as isize);
    }

    fn keywords_changed(&mut self, node_count_change: isize) {
        self.compiled.take();
        *self.work_since_change.get_mut() = 0;
        if let Some(node_count) = self.node_count.get_mut() {
            *node_count = node_count.saturating_add_signed(node_count_change);
        }
    }

    /// Removes a keyword from the processor.
    ///
    /// This method removes the specified keyword from the trie data structure.
    /// If the keyword is found and removed, it returns true. Otherwise, it returns false.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The keyword to remove.
    ///
    /// # Returns
    ///
    /// `true` if the keyword was found and removed, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", None);
    /// assert!(processor.remove_keyword("rust"));
    /// assert!(!processor.remove_keyword("python")); // Not in the trie
    /// ```
    pub fn remove_keyword(&mut self, keyword: &str) -> bool {
        let keyword = if self.case_sensitive {
            keyword.to_string()
        } else {
            keyword.to_lowercase()
        };

        let mut removed_nodes = 0;
        let removed = Self::remove_keyword_helper(keyword.as_bytes(), &mut self.root, &mut removed_nodes);
        if removed {
            self.keywords_changed(-(removed_nodes as isize));
        }
        removed
    }

    /// Helper function for removing a keyword from the trie.
    ///
    /// This method uses a recursive approach to traverse the trie and remove the keyword.
    /// Here's how it works:
    ///
    /// 1. If the keyword is empty (base case):
    ///    a. If the current node has a keyword, remove it and return true.
    ///    b. Otherwise, return false (keyword not found).
    /// 2. If the keyword is not empty:
    ///    a. Get the first character of the keyword.
    ///    b. If there's a child node for this character:
    ///       - Recursively call remove_keyword_helper with the rest of the keyword.
    ///       - If the recursive call returned true and the child node is now empty:
    ///         * Remove the child node.
    ///       - Return the result of the recursive call.
    ///    c. If there's no child node for this character, return false (keyword not found).
    ///
    /// This approach ensures that we remove the keyword and any unnecessary nodes,
    /// while keeping the rest of the trie intact.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The remaining part of the keyword to remove.
    /// * `node` - The current node in the trie.
    ///
    /// # Returns
    ///
    /// `true` if the keyword was found and removed, `false` otherwise.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(m), where m is the length of the keyword.
    /// * Space complexity: O(m) due to the recursion stack.
    ///
    fn remove_keyword_helper(keyword: &[u8], node: &mut TrieNode, removed_nodes: &mut usize) -> bool {
        if let Some((&first_byte, rest)) = keyword.split_first() {
            let key = first_byte as char;
            if let Some(child) = node.children.get_mut(&key) {
                let removed = Self::remove_keyword_helper(rest, child, removed_nodes);
                if removed && child.children.is_empty() && child.keyword.is_none() {
                    node.children.remove(&key);
                    *removed_nodes += 1;
                }
                removed
            } else {
                false
            }
        } else {
            node.keyword.take().is_some()
        }
    }

    /// Adds multiple keywords from a list to the processor.
    ///
    /// This method iterates through the provided list of keywords and their
    /// optional clean names, adding each to the trie structure.
    ///
    /// # Arguments
    ///
    /// * `keywords` - A slice of tuples, each containing a keyword and its optional clean name.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// let keywords = vec![
    ///     ("rust", Some("Rust Programming Language")),
    ///     ("python", None),
    /// ];
    /// processor.add_keywords_from_list(&keywords);
    /// ```
    pub fn add_keywords_from_list(&mut self, keywords: &[(&str, Option<&str>)]) {
        for (keyword, clean_name) in keywords {
            self.add_keyword(keyword, *clean_name);
        }
    }

    /// Removes multiple keywords from the processor.
    ///
    /// This method attempts to remove each keyword in the provided list from the trie structure.
    /// It returns a vector of boolean values indicating whether each keyword was successfully removed.
    ///
    /// # Arguments
    ///
    /// * `keywords` - A slice of keyword strings to remove.
    ///
    /// # Returns
    ///
    /// A `Vec<bool>` where each element corresponds to a keyword in the input slice.
    /// `true` indicates the keyword was found and removed, `false` indicates it was not found.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", None);
    /// processor.add_keyword("python", None);
    ///
    /// let results = processor.remove_keywords_from_list(&["rust", "java"]);
    /// assert_eq!(results, vec![true, false]); // "rust" removed, "java" not found
    /// ```
    pub fn remove_keywords_from_list(&mut self, keywords: &[&str]) -> Vec<bool> {
        keywords
            .iter()
            .map(|keyword| self.remove_keyword(keyword))
            .collect()
    }

    /// Retrieves all keywords and their clean names from the processor.
    ///
    /// This method traverses the entire trie structure to collect all stored keywords
    /// and their corresponding clean names.
    ///
    /// # Returns
    ///
    /// A `HashMap` where keys are the original keywords and values are their clean names.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Lang"));
    /// processor.add_keyword("python", None);
    ///
    /// let all_keywords = processor.get_all_keywords();
    /// assert_eq!(all_keywords.get("rust"), Some(&"Rust Lang".to_string()));
    /// assert_eq!(all_keywords.get("python"), Some(&"python".to_string()));
    /// ```
    pub fn get_all_keywords(&self) -> HashMap<String, String> {
        let mut keywords = HashMap::new();
        self.collect_keywords(&self.root, &mut Vec::new(), &mut keywords);
        keywords
    }

    /// Helper method for get_all_keywords. Recursively collects keywords from the trie.
    ///
    /// This method performs a depth-first traversal of the trie, building up the keyword
    /// strings and adding them to the HashMap when a keyword node is encountered.
    ///
    /// # Arguments
    ///
    /// * `node` - The current TrieNode being examined.
    /// * `prefix` - The string built up from the root to the current node.
    /// * `keywords` - A mutable reference to the HashMap where collected keywords are stored.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(n), where n is the total number of characters in all keywords.
    /// * Space complexity: O(m) where m is the length of the longest keyword (due to recursion stack).
    ///
    fn collect_keywords(
        &self,
        node: &TrieNode,
        prefix: &mut Vec<u8>,
        keywords: &mut HashMap<String, String>,
    ) {
        if let Some(clean_name) = &node.keyword {
            let keyword = String::from_utf8_lossy(prefix).into_owned();
            keywords.insert(keyword, clean_name.clone());
        }
        for (&byte, child) in &node.children {
            prefix.push(byte as u8);
            self.collect_keywords(child, prefix, keywords);
            prefix.pop();
        }
    }

    /// Sets the non-word boundaries to a new set of characters.
    ///
    /// Non-word boundaries are characters that are considered part of a word.
    /// This method replaces the existing set of non-word boundaries with a new set.
    ///
    /// # Arguments
    ///
    /// * `boundaries` - A slice of characters to be set as non-word boundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.set_non_word_boundaries(&['a', 'b', 'c', '1', '2', '3']);
    /// ```
    pub fn set_non_word_boundaries(&mut self, boundaries: &[char]) {
        self.non_word_boundaries = CompactCharSet::new();
        self.non_word_boundaries.extend(boundaries.iter().copied());
    }

    /// Adds a single character as a non-word boundary.
    ///
    /// This method adds a new character to the existing set of non-word boundaries.
    /// Non-word boundaries are characters that are considered part of a word.
    ///
    /// # Arguments
    ///
    /// * `boundary` - The character to be added as a non-word boundary.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_non_word_boundary('_');
    /// ```

    pub fn add_non_word_boundary(&mut self, boundary: char) {
        self.non_word_boundaries.insert(boundary);
    }

    /// Returns the total number of keywords stored in the processor.
    ///
    /// # Returns
    ///
    /// The number of keywords in the processor.
    pub fn len(&self) -> usize {
        self.count_keywords(&self.root)
    }

    /// Helper method to recursively count keywords in the trie.
    ///
    /// This method performs a depth-first traversal of the trie,
    /// counting nodes that represent complete keywords.
    ///
    /// # Arguments
    ///
    /// * `node` - The current TrieNode being examined.
    ///
    /// # Returns
    ///
    /// The number of keywords in the subtree rooted at `node`.
    fn count_keywords(&self, node: &TrieNode) -> usize {
        let mut count = if node.keyword.is_some() { 1 } else { 0 };
        for child in node.children.values() {
            count += self.count_keywords(child);
        }
        count
    }

    /// Retrieves the clean name for a given keyword.
    ///
    /// This method searches the trie for an exact match of the given keyword
    /// and returns its associated clean name if found.
    ///
    /// # Arguments
    ///
    /// * `keyword` - The keyword to look up.
    ///
    /// # Returns
    ///
    /// An `Option` containing the clean name if the keyword exists, or `None` if it doesn't.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(m), where m is the length of the keyword.
    /// * Space complexity: O(1)
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Programming Language"));
    /// assert_eq!(processor.get_keyword("rust"), Some("Rust Programming Language".to_string()));
    /// assert_eq!(processor.get_keyword("python"), None);
    /// ```
    pub fn get_keyword(&self, keyword: &str) -> Option<String> {
        let keyword = self.normalize_case(keyword);
        let mut current_node = &self.root;
        for byte in keyword.bytes() {
            current_node = current_node.children.get(&(byte as char))?;
        }
        current_node.keyword.clone()
    }

    /// Extracts keywords from the given text.
    ///
    /// This method searches for all keywords in the given text and returns them.
    /// It uses the trie data structure for efficient matching and respects word boundaries.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to search for keywords.
    /// * `threshold` - Threshold for the Normalized Levenshtein distance for fuzzy matching.
    ///                 If set to 1.0, only exact matches are returned. Optional, defaults to 1.0.
    ///
    /// # Returns
    ///
    /// A vector of `KeywordMatch` objects, each containing the keyword, similarity, and span.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(n * m), where n is the length of the text and m is the average length of keywords.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Lang"));
    /// let matches = processor.extract_keywords("I love rust programming", None);
    /// assert_eq!(matches.len(), 1);
    /// assert_eq!(matches[0].keyword, "Rust Lang");
    /// ```
    pub fn extract_keywords<'a>(
        &'a self,
        text: &'a str,
        threshold: Option<f32>,
    ) -> Vec<KeywordMatch<'a>> {
        // Use the provided threshold or default to 1.0 (exact matches only)
        let threshold = threshold.unwrap_or(1.0);
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Threshold must be between 0 and 1"
        );

        let original = text;
        let text = self.normalize_case(original);
        let mut matches = Vec::new();
        let trie = self.compiled_trie(text.len(), threshold);
        self.scan_with(trie, &text, 0..text.len(), threshold, &mut matches);
        self.map_to_original_offsets(original, &mut matches);
        matches
    }

    /// Extracts keywords from the given text using parallel processing.
    ///
    /// This method is similar to `extract_keywords`, but it uses parallel processing to
    /// improve performance on longer texts. It divides the text into chunks and processes
    /// them concurrently.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to search for keywords.
    /// * `threshold` - Threshold for the Normalized Levenshtein distance for fuzzy matching.
    ///                 If set to 1.0, only exact matches are returned. Optional, defaults to 1.0.
    ///
    /// # Returns
    ///
    /// A vector of `KeywordMatch` objects, each containing the keyword, similarity, and span.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(n * m / p), where n is the length of the text, m is the average length of keywords,
    ///   and p is the number of parallel threads.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Lang"));
    /// let matches = processor.extract_keywords_parallel("I love rust programming", None);
    /// assert_eq!(matches.len(), 1);
    /// assert_eq!(matches[0].keyword, "Rust Lang");
    /// ```
    pub fn extract_keywords_parallel<'a>(
        &'a self,
        text: &'a str,
        threshold: Option<f32>,
    ) -> Vec<KeywordMatch<'a>> {
        let threshold = threshold.unwrap_or(1.0);
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Threshold must be between 0 and 1"
        );

        let original = text;
        let text = self.normalize_case(original);
        let trie = self.compiled_trie(text.len(), threshold);
        let chunk_size = (text.len() / rayon::current_num_threads()).max(1000);

        let mut matches: Vec<KeywordMatch> = (0..text.len())
            .into_par_iter()
            .step_by(chunk_size)
            .flat_map(|start| {
                let end = (start + chunk_size).min(text.len());
                let mut local_matches = Vec::new();
                self.scan_with(trie, &text, start..end, threshold, &mut local_matches);
                local_matches
            })
            .collect();
        self.map_to_original_offsets(original, &mut matches);
        matches
    }

    /// Extracts keywords from multiple texts in parallel.
    ///
    /// This method processes multiple texts concurrently, extracting keywords from each.
    /// It's useful for batch processing of multiple documents.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of texts to search for keywords.
    /// * `threshold` - Threshold for the Normalized Levenshtein distance for fuzzy matching.
    ///                 If set to 1.0, only exact matches are returned. Optional, defaults to 1.0.
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector contains `KeywordMatch` objects for one text.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(N * n * m / p), where N is the number of texts, n is the average length of texts,
    ///   m is the average length of keywords, and p is the number of parallel threads.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Lang"));
    /// let texts = vec!["I love rust", "Python is cool too"];
    /// let results = processor.parallel_extract_keywords_from_texts(&texts, None);
    /// assert_eq!(results.len(), 2);
    /// assert_eq!(results[0].len(), 1);  // One match in the first text
    /// assert_eq!(results[1].len(), 0);  // No matches in the second text
    /// ```
    pub fn parallel_extract_keywords_from_texts<'a, T>(
        &'a self,
        texts: &'a [T],
        threshold: Option<f32>,
    ) -> Vec<Vec<KeywordMatch<'a>>>
    where
        // allows us to use various types that can be converted to string slices
        T: AsRef<str> + Sync,
    {
        texts
            .par_iter()
            .map(|text| self.extract_keywords(text.as_ref(), threshold))
            .collect()
    }

    /// Scans with the compiled trie when there is one, else with the editable trie.
    fn scan_with<'a>(
        &'a self,
        trie: Option<&'a CompiledTrie>,
        text: &str,
        positions: std::ops::Range<usize>,
        threshold: f32,
        matches: &mut Vec<KeywordMatch<'a>>,
    ) {
        match trie {
            Some(trie) => self.scan(trie, text, positions, threshold, matches),
            None => self.scan(&self.root, text, positions, threshold, matches),
        }
    }

    /// Appends the matches that start inside `positions` to `matches`; a match may run past its end.
    fn scan<'a, T: TrieView<'a>>(
        &self,
        trie: T,
        text: &str,
        positions: std::ops::Range<usize>,
        threshold: f32,
        matches: &mut Vec<KeywordMatch<'a>>,
    ) {
        let text_bytes = text.as_bytes();
        let is_ascii_word_byte: [bool; 128] =
            std::array::from_fn(|byte| self.is_non_word_boundary(byte as u8 as char));
        // Whether the character at byte `i` is a word character, and its length in bytes
        let word_char_at = |i: usize| match text_bytes[i] {
            byte if byte < 128 => (is_ascii_word_byte[byte as usize], 1),
            _ => {
                let c = text[i..].chars().next().unwrap();
                (self.is_non_word_boundary(c), c.len_utf8())
            }
        };
        let is_word_char_before = |i: usize| match text_bytes[i - 1] {
            byte if byte < 128 => is_ascii_word_byte[byte as usize],
            _ => self.is_non_word_boundary(text[..i].chars().next_back().unwrap()),
        };

        let mut fuzzy_scratch = FuzzyScratch::default();
        let mut i = positions.start;
        'scan: while i < positions.end {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            // A keyword can only start at the text start or after a non-word character
            if i > 0 && is_word_char_before(i) {
                loop {
                    let Some(&byte) = text_bytes[..positions.end].get(i) else {
                        break 'scan;
                    };
                    // Runs once per text byte, so ASCII stays inline
                    let (is_word, char_len) = match byte {
                        byte if byte < 128 => (is_ascii_word_byte[byte as usize], 1),
                        _ => word_char_at(i),
                    };
                    i += char_len;
                    if !is_word {
                        continue 'scan;
                    }
                }
            }
            // Choose between fuzzy and exact matching based on threshold
            let (found, length, similarity) = if threshold < 1.0 {
                fuzzy::best_fuzzy_prefix(trie, &text[i..], threshold, &mut fuzzy_scratch)
            } else {
                match trie.longest_prefix(&text_bytes[i..]) {
                    Some((keyword, length)) => (Some(keyword), length, 1.0),
                    None => (None, 0, 0.0),
                }
            };

            if let Some(keyword) = found {
                let start = if is_whitespace(text_bytes[i]) { i + 1 } else { i };
                if threshold == 1.0 {
                    // For ensuring that we only match complete words, not partial words or substrings within larger words.
                    // This is true if we're at the end of the text or the next character is not a word character
                    if i + length == text_bytes.len() || !word_char_at(i + length).0 {
                        matches.push(KeywordMatch {
                            keyword,
                            similarity: round(similarity, 2),
                            start,
                            end: i + length,
                        });
                        i += length;
                        continue;
                    }
                } else {
                    // For fuzzy matching, accept a word boundary within the next 3 characters
                    let end = if is_whitespace(text_bytes[i + length - 1]) {
                        i + length - 1
                    } else {
                        i + length
                    };
                    let is_boundary_in_next_x_chars = text[end..]
                        .chars()
                        .take(3)
                        .any(|c| !self.is_non_word_boundary(c));
                    // A lone whitespace match trims to an inverted span, which is no match.
                    if start <= end && (is_boundary_in_next_x_chars || end == text_bytes.len()) {
                        matches.push(KeywordMatch {
                            keyword,
                            similarity: round(similarity, 2),
                            start,
                            end,
                        });
                        i += length;
                        continue;
                    }
                }
            }
            // If no keyword found or not at a word boundary, move to the next character
            i += 1;
        }
    }

    /// Matches are found in the lowercased text; when lowercasing changed its length,
    /// moves each match onto the original characters it came from.
    fn map_to_original_offsets(&self, original: &str, matches: &mut [KeywordMatch]) {
        if self.case_sensitive || matches.is_empty() {
            return;
        }
        let Some(char_starts) = lowercase_char_starts(original) else {
            return;
        };
        for keyword_match in matches {
            let start = char_starts[keyword_match.start];
            keyword_match.end = if keyword_match.end == keyword_match.start {
                start
            } else {
                let last_char = char_starts[keyword_match.end - 1];
                last_char + original[last_char..].chars().next().map_or(0, char::len_utf8)
            };
            keyword_match.start = start;
        }
    }

    fn normalize_case<'t>(&self, text: &'t str) -> Cow<'t, str> {
        if self.case_sensitive {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(text.to_lowercase())
        }
    }

    /// Returns the compiled trie once bytes scanned since the last edit (fuzzy weighted 64x)
    /// reach 16x the node count; until then matching uses the editable trie.
    fn compiled_trie(&self, text_len: usize, threshold: f32) -> Option<&CompiledTrie> {
        if let Some(trie) = self.compiled.get() {
            return Some(trie);
        }
        // Fuzzy matching costs far more per byte than exact matching.
        let work = if threshold < 1.0 { text_len * 64 } else { text_len };
        let total_work = self.work_since_change.fetch_add(work, Ordering::Relaxed) + work;
        let node_count = *self.node_count.get_or_init(|| self.root.count_nodes());
        (total_work >= node_count * 16).then(|| self.compiled.get_or_init(|| CompiledTrie::new(&self.root)))
    }

    /// Replaces keywords in the given text with their clean names.
    ///
    /// This method searches for keywords in the text and replaces them with their
    /// corresponding clean names. It can perform either exact or fuzzy matching
    /// based on the provided threshold.
    ///
    /// # Arguments
    ///
    /// * `text` - The text in which to replace keywords.
    /// * `threshold` - Threshold for the Normalized Levenshtein distance for fuzzy matching.
    ///                 If set to 1.0, only exact matches are replaced. Optional, defaults to 1.0.
    ///
    /// # Returns
    ///
    /// A String with keywords replaced by their clean names.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(n * m), where n is the length of the text and m is the average length of keywords.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Programming Language"));
    /// let result = processor.replace_keywords("I love rust programming", None);
    /// assert_eq!(result, "I love Rust Programming Language programming");
    /// ```
    pub fn replace_keywords(&self, text: &str, threshold: Option<f32>) -> String {
        let mut result = String::with_capacity(text.len());
        let mut copied_up_to = 0;
        for keyword_match in self.extract_keywords(text, threshold) {
            // Two matches inside one character that lowercases to several can overlap.
            let start = keyword_match.start.max(copied_up_to);
            result.push_str(&text[copied_up_to..start]);
            result.push_str(keyword_match.keyword);
            copied_up_to = keyword_match.end.max(start);
        }
        result.push_str(&text[copied_up_to..]);
        result
    }

    /// Replaces keywords in multiple texts with their clean names, processing in parallel.
    ///
    /// # Example
    ///
    /// ```
    /// use blitztext::KeywordProcessor;
    ///
    /// let mut processor = KeywordProcessor::new();
    /// processor.add_keyword("rust", Some("Rust Programming Language"));
    /// processor.add_keyword("python", Some("Python Programming Language"));
    ///
    /// let texts = vec![
    ///     "I love rust programming",
    ///     "Python is also great",
    ///     "Neither russt nor pithon are mentioned here"
    /// ];
    ///
    /// let results = processor.parallel_replace_keywords_from_texts(&texts, None);
    /// println!("{:?}", results);
    ///
    /// assert_eq!(results[0], "I love Rust Programming Language programming");
    /// assert_eq!(results[1], "Python Programming Language is also great");
    /// assert_eq!(results[2], "Neither russt nor pithon are mentioned here");
    /// ```
    pub fn parallel_replace_keywords_from_texts<T>(
        &self,
        texts: &[T],
        threshold: Option<f32>,
    ) -> Vec<String>
    where
        T: AsRef<str> + Sync,
    {
        texts
            .par_iter()
            .map(|text| self.replace_keywords(text.as_ref(), threshold))
            .collect()
    }

    /// Checks if a character is considered a non-word boundary.
    ///
    /// Non-word boundaries are characters that are considered part of a word.
    /// This method is used internally to determine word boundaries during keyword matching.
    ///
    /// # Arguments
    ///
    /// * `ch` - The character to check.
    ///
    /// # Returns
    ///
    /// `true` if the character is a non-word boundary, `false` otherwise.
    ///
    /// # Note
    ///
    /// If `allow_overlaps` is set to true, this method always returns `false`,
    /// effectively disabling word boundary checks.
    fn is_non_word_boundary(&self, ch: char) -> bool {
        if self.allow_overlaps {
            return false;
        }
        if self.non_word_boundaries.contains(ch) {
            return true;
        }
        // Text is lowercased before matching, so an uppercase boundary covers its lowercase form.
        let mut uppercase = ch.to_uppercase();
        match (self.case_sensitive, uppercase.next(), uppercase.next()) {
            (false, Some(upper), None) if upper != ch => self.non_word_boundaries.contains(upper),
            _ => false,
        }
    }

    /// Calculates the approximate memory usage of the trie in bytes.
    ///
    /// This method provides an estimate of the memory used by the trie structure,
    /// including all nodes and the strings stored in them, plus the compiled
    /// matching copy once it has been built.
    ///
    /// # Returns
    ///
    /// The estimated memory usage in bytes.
    pub fn calculate_memory_usage(&self) -> usize {
        let trie_memory = self.calculate_node_memory(&self.root);

        let compiled_memory = self.compiled.get().map_or(0, CompiledTrie::memory_usage);
        // Add size of KeywordProcessor struct itself
        trie_memory + compiled_memory + size_of::<KeywordProcessor>()
    }

    fn calculate_node_memory(&self, node: &TrieNode) -> usize {
        let mut memory = size_of::<TrieNode>();

        // Add memory for HashMap
        memory += size_of::<HashMap<char, Box<TrieNode>>>();
        memory += node.children.capacity() * (size_of::<char>() + size_of::<Box<TrieNode>>());

        // Add memory for keyword if present
        if let Some(ref keyword) = node.keyword {
            memory += keyword.capacity();
        }

        // Recursively calculate memory for child nodes
        for child in node.children.values() {
            memory += self.calculate_node_memory(child);
        }

        memory
    }
}

/// For text whose lowercase form has a different byte length, maps every byte offset
/// of `text.to_lowercase()` to the start of the original character it came from.
fn lowercase_char_starts(text: &str) -> Option<Vec<usize>> {
    let lowercase_len = |c: char| c.to_lowercase().map(char::len_utf8).sum::<usize>();
    if text.is_ascii() || text.chars().all(|c| lowercase_len(c) == c.len_utf8()) {
        return None;
    }
    let mut starts = Vec::with_capacity(text.len() + 1);
    for (offset, c) in text.char_indices() {
        starts.extend(std::iter::repeat_n(offset, lowercase_len(c)));
    }
    starts.push(text.len());
    Some(starts)
}

/// A struct representing a keyword match in a text.
#[derive(Debug, PartialEq)]
pub struct KeywordMatch<'a> {
    pub keyword: &'a str,
    pub similarity: f32,
    pub start: usize,
    pub end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_keyword() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));
        // println!("m  {:?}", (char::MAX as usize) / 64 + 1);

        assert!(kp.remove_keyword("rust"));
        assert!(!kp.remove_keyword("rust")); // Already removed
        assert!(kp.remove_keyword("programming"));
        assert!(!kp.remove_keyword("coding")); // Not a keyword, but a clean name
    }

    #[test]
    fn test_extract_keywords() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer", None);
        kp.add_keyword("amming", None);

        let text = "I love rust programming and computer science.";
        let keywords = kp.extract_keywords(text, None);

        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 7,
                    end: 11,
                },
                KeywordMatch {
                    keyword: "coding",
                    similarity: 1.0,
                    start: 12,
                    end: 23,
                },
                KeywordMatch {
                    keyword: "computer",
                    similarity: 1.0,
                    start: 28,
                    end: 36,
                },
            ]
        );

        kp.add_keyword("寿司", Some("Sushi"));
        kp.add_keyword("刺身", Some("Sashimi"));

        let text = "私は寿司と刺身が好きです。";

        let matches = kp.extract_keywords(text, None);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].keyword, "Sushi");
        assert_eq!(matches[1].keyword, "Sashimi");
    }

    #[test]
    fn text_extract_keywords_with_overlaps() {
        let mut kp = KeywordProcessor::with_options(false, true);
        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer", None);
        kp.add_keyword("amming", None);

        let text = "Ilove rust programming and computerscience.";
        let keywords = kp.extract_keywords(text, None);
        // println!("{:?}", keywords);
        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 6,
                    end: 10,
                },
                KeywordMatch {
                    keyword: "coding",
                    similarity: 1.0,
                    start: 11,
                    end: 22,
                },
                KeywordMatch {
                    keyword: "computer",
                    similarity: 1.0,
                    start: 27,
                    end: 35,
                },
            ]
        );
    }

    #[test]
    fn test_extract_keywords_parallel() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer", None);
        kp.add_keyword("amming", None);

        let text = "I love rust programming and computer science.";
        let mut keywords = kp.extract_keywords_parallel(text, None);
        keywords.sort_by(|a, b| a.keyword.cmp(&b.keyword));
        // println!("{:?}", keywords);

        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "coding",
                    similarity: 1.0,
                    start: 12,
                    end: 23,
                },
                KeywordMatch {
                    keyword: "computer",
                    similarity: 1.0,
                    start: 28,
                    end: 36,
                },
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 7,
                    end: 11,
                },
            ]
        );
    }

    #[test]
    fn test_parallel_extract_keywords_from_texts() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer", None);
        kp.add_keyword("amming", None);

        let texts = [
            "I love rust programming and computer science.",
            "I am a programmerg and I love coding.",
        ];

        let results = kp.parallel_extract_keywords_from_texts(&texts, Some(0.7));
        // println!("{:?}", results);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0],
            vec![
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 7,
                    end: 11,
                },
                KeywordMatch {
                    keyword: "coding",
                    similarity: 1.0,
                    start: 12,
                    end: 23,
                },
                KeywordMatch {
                    keyword: "computer",
                    similarity: 1.0,
                    start: 28,
                    end: 36,
                },
            ]
        );
        assert_eq!(
            results[1],
            vec![KeywordMatch {
                keyword: "coding",
                similarity: 0.82,
                start: 7,
                end: 18,
            },]
        );
    }

    #[test]
    fn test_fuzzy_search() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("hello", None);
        kp.add_keyword("world", None);

        // Exact match
        let result = kp.extract_keywords("hello", None);
        // println!("{:?}", result);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result,
            vec![KeywordMatch {
                keyword: "hello",
                similarity: 1.0,
                start: 0,
                end: 5,
            }]
        );

        // Fuzzy match with threshold
        let result = kp.extract_keywords("wurld", Some(0.5));
        // println!("res: {:?}", result);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result,
            vec![KeywordMatch {
                keyword: "world",
                similarity: 0.8,
                start: 0,
                end: 5,
            }]
        );

        // Fuzzy match that doesn't meet the threshold
        let result = kp.extract_keywords("worlllds", Some(1.0));
        assert!(result.is_empty());

        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", None);
        kp.add_keyword("love", None);
        kp.add_keyword("programming", Some("coding"));

        let text = "I genuinely ove rus progrming and computer science. Rustylang is great!";
        let keywords = kp.extract_keywords(text, Some(0.6));
        // println!("fuzzzy {:?}", keywords);
        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "love",
                    similarity: 0.75,
                    start: 12,
                    end: 15,
                },
                KeywordMatch {
                    keyword: "rust",
                    similarity: 0.75,
                    start: 16,
                    end: 19,
                },
                KeywordMatch {
                    keyword: "coding",
                    similarity: 0.82,
                    start: 20,
                    end: 29,
                },
            ]
        );

        // Multiple matches with insertion, deletion, and substitution
        kp.add_keyword("programming and computer science", Some("tech"));
        kp.add_keyword("rusty lang", None);
        // kp.add_keyword("rusty lang", None);
        let text = "I love rust programYing and computter science. Rusty-lang is great!";
        let keywords = kp.extract_keywords(text, Some(0.75));
        // let keywords = kp.extract_keywords(text, Some(0.8));
        // keywords.sort_by(|a, b| b.keyword.cmp(&a.keyword));
        // println!("{:?}", keywords);
        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "love",
                    similarity: 1.0,
                    start: 2,
                    end: 6,
                },
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 7,
                    end: 11,
                },
                KeywordMatch {
                    keyword: "tech",
                    similarity: 0.94,
                    start: 12,
                    end: 45,
                },
                KeywordMatch {
                    keyword: "rusty lang",
                    similarity: 0.82,
                    start: 47,
                    end: 57,
                },
            ]
        );
    }

    #[test]
    fn test_replace_keywords() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", Some("Rust"));
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer science", Some("CS"));

        let text = "I love rust programming and computer science.";
        let result = kp.replace_keywords(text, None);
        assert_eq!(result, "I love Rust coding and CS.");

        // Test with fuzzy matching
        let text = "I love rst programing and computter science.";
        // let result = kp.replace_keywords(text, Some(0.4));
        let result = kp.replace_keywords(text, Some(0.7));
        assert_eq!(result, "I love Rust coding and CS.");
    }

    #[test]
    fn test_parallel_replace_keywords_from_texts() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("rust", Some("Rust"));
        kp.add_keyword("programming", Some("coding"));
        kp.add_keyword("computer science", Some("CS"));

        let texts = [
            "I love rust programming and computer science.",
            "I am a programmer and I love coding.",
        ];

        let results = kp.parallel_replace_keywords_from_texts(&texts, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "I love Rust coding and CS.");
        assert_eq!(results[1], "I am a programmer and I love coding.");
    }

    #[test]
    fn test_add_keywords_from_list() {
        let mut kp = KeywordProcessor::new();
        kp.add_keywords_from_list(&[
            ("rust", Some("Rust")),
            ("programming", Some("coding")),
            ("computer", None),
        ]);

        assert_eq!(kp.get_keyword("rust"), Some("Rust".to_string()));
        assert_eq!(kp.get_keyword("programming"), Some("coding".to_string()));
        assert_eq!(kp.get_keyword("computer"), Some("computer".to_string()));
    }

    #[test]
    fn test_remove_keywords_from_list() {
        let mut kp = KeywordProcessor::new();
        kp.add_keywords_from_list(&[
            ("rust", Some("Rust")),
            ("programming", Some("coding")),
            ("computer", None),
        ]);

        let results = kp.remove_keywords_from_list(&["rust", "programming", "nonexistent"]);
        assert_eq!(results, vec![true, true, false]);

        assert_eq!(kp.get_keyword("rust"), None);
        assert_eq!(kp.get_keyword("programming"), None);
        assert_eq!(kp.get_keyword("computer"), Some("computer".to_string()));
    }

    #[test]
    fn test_get_all_keywords() {
        let mut kp = KeywordProcessor::new();
        kp.add_keywords_from_list(&[
            ("rust", Some("Rust")),
            ("programming", Some("coding")),
            ("computer", None),
        ]);

        let all_keywords = kp.get_all_keywords();
        assert_eq!(all_keywords.len(), 3);
        assert_eq!(all_keywords.get("rust"), Some(&"Rust".to_string()));
        assert_eq!(all_keywords.get("programming"), Some(&"coding".to_string()));
        assert_eq!(all_keywords.get("computer"), Some(&"computer".to_string()));
    }

    #[test]
    fn test_get_keyword() {
        let mut kp = KeywordProcessor::new();
        kp.add_keywords_from_list(&[
            ("rust", Some("Rust")),
            ("programming", Some("coding")),
            ("computer", None),
        ]);

        assert_eq!(kp.get_keyword("rust"), Some("Rust".to_string()));
        assert_eq!(kp.get_keyword("PROGRAMMING"), Some("coding".to_string())); // Case-insensitive
        assert_eq!(kp.get_keyword("computer"), Some("computer".to_string()));
        assert_eq!(kp.get_keyword("nonexistent"), None);
    }

    #[test]
    fn test_set_non_word_boundaries() {
        let mut kp = KeywordProcessor::new();

        // Test with array of chars
        kp.set_non_word_boundaries(&['a', 'b', 'c', '1', '2', '3']);
        assert!(kp.is_non_word_boundary('a'));
        assert!(kp.is_non_word_boundary('3'));
        assert!(!kp.is_non_word_boundary('d'));
    }

    #[test]
    fn test_add_non_word_boundary() {
        let mut kp = KeywordProcessor::new();

        kp.add_non_word_boundary('$');
        assert!(kp.is_non_word_boundary('$'));
        assert!(!kp.is_non_word_boundary('%'));

        kp.add_non_word_boundary('%');
        assert!(kp.is_non_word_boundary('$'));
        assert!(kp.is_non_word_boundary('%'));
    }

    #[test]
    fn test_keywords_length() {
        let mut kp = KeywordProcessor::new();
        assert_eq!(kp.len(), 0);

        kp.add_keyword("rust", None);
        assert_eq!(kp.len(), 1);

        kp.add_keyword("programming", Some("coding"));
        assert_eq!(kp.len(), 2);

        kp.add_keyword("rust", Some("Rust")); // Overwriting existing keyword
        assert_eq!(kp.len(), 2);

        kp.remove_keyword("rust");
        assert_eq!(kp.len(), 1);

        kp.remove_keyword("nonexistent");
        assert_eq!(kp.len(), 1);
    }

    #[test]
    fn test_extract_keywords_with_custom_boundaries() {
        let mut kp = KeywordProcessor::new();
        // kp.set_non_word_boundaries(&['a', 'b', 'c', '1', '2', '3']);

        kp.add_keyword("rust", None);
        kp.add_keyword("programming", Some("coding"));

        let text = "I-love-rust-programming-and-1coding2";
        let keywords = kp.extract_keywords(text, None);
        // println!("{:?}", keywords);

        assert_eq!(
            keywords,
            vec![
                KeywordMatch {
                    keyword: "rust",
                    similarity: 1.0,
                    start: 7,
                    end: 11,
                },
                KeywordMatch {
                    keyword: "coding",
                    similarity: 1.0,
                    start: 12,
                    end: 23
                },
            ]
        );

        kp.add_non_word_boundary('-');
        let keywords = kp.extract_keywords(text, None);
        // println!("{:?}", keywords);
        assert_eq!(keywords.len(), 0);
    }

    #[test]
    fn compiled_trie_matches_like_the_editable_trie() {
        use rand::{rngs::StdRng, RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        let pieces = ["a", "b", "ab", "ba", "c", "é", "日", "😀", " ", " ", "-", "x"];
        let random_string = |rng: &mut StdRng, max_pieces: usize| -> String {
            (0..rng.random_range(0..=max_pieces))
                .map(|_| pieces[rng.random_range(0..pieces.len())])
                .collect()
        };
        for case in 0..300 {
            let mut kp = KeywordProcessor::with_options(case % 3 == 0, case % 5 == 0);
            for _ in 0..rng.random_range(1..40) {
                let keyword = random_string(&mut rng, 4);
                let clean_name = random_string(&mut rng, 3);
                kp.add_keyword(&keyword, rng.random_bool(0.5).then_some(clean_name.as_str()));
            }
            let compiled = CompiledTrie::new(&kp.root);
            for _ in 0..20 {
                let text = random_string(&mut rng, 30);
                for threshold in [1.0, 0.95, 0.8, 0.7, 0.6, 0.5] {
                    let (mut expected, mut actual) = (Vec::new(), Vec::new());
                    kp.scan(&kp.root, &text, 0..text.len(), threshold, &mut expected);
                    kp.scan(&compiled, &text, 0..text.len(), threshold, &mut actual);
                    assert_eq!(actual, expected, "text {text:?} threshold {threshold}");
                }
            }
        }
    }

    #[test]
    fn non_ascii_non_word_boundary_joins_words() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("caf", None);
        assert_eq!(kp.extract_keywords("caf café", None).len(), 2);
        kp.add_non_word_boundary('é');
        let matches = kp.extract_keywords("caf café", None);
        assert_eq!((matches.len(), matches[0].start, matches[0].end), (1, 0, 3));
    }

    #[test]
    fn uppercase_boundary_applies_in_case_insensitive_mode() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("caf", None);
        kp.add_non_word_boundary('É');
        assert_eq!(kp.extract_keywords("CAFÉ", None), vec![]);
        assert_eq!(kp.extract_keywords("CAF.", None).len(), 1);
    }

    #[test]
    fn offsets_point_into_original_text_when_lowercase_is_longer() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("istanbul", Some("IST"));
        let text = "İstanbul istanbul";
        let matches = kp.extract_keywords(text, None);
        assert_eq!(&text[matches[0].start..matches[0].end], "istanbul");
    }

    #[test]
    fn fuzzy_tie_goes_to_smallest_clean_name() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("cat", None);
        kp.add_keyword("car", None);
        let matches = kp.extract_keywords("cab", Some(0.6));
        assert_eq!((matches[0].keyword, matches[0].similarity), ("car", 0.67));
    }

    #[test]
    fn fuzzy_matches_a_missing_last_letter_at_text_end() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("python", None);
        let matches = kp.extract_keywords("I use pytho", Some(0.8));
        assert_eq!((matches[0].keyword, matches[0].start, matches[0].end), ("python", 6, 11));
    }

    #[test]
    fn fuzzy_match_leaves_trailing_punctuation_out() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("python", None);
        assert_eq!(kp.replace_keywords("I like pytho.", Some(0.8)), "I like python.");
    }

    #[test]
    fn fuzzy_whitespace_only_match_is_dropped() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("a ", Some("X"));
        for text in ["Klong  ", "klong  "] {
            assert!(kp.extract_keywords(text, Some(0.5)).iter().all(|m| m.start <= m.end));
            assert_eq!(kp.replace_keywords(text, Some(0.5)), text);
        }
    }

    #[test]
    fn fuzzy_counts_a_wrong_accent_as_one_edit() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("cafe", None);
        let matches = kp.extract_keywords("café", Some(0.75));
        assert_eq!((matches[0].similarity, matches[0].end), (0.75, 5));
    }

    fn processor_with_long_clean_name() -> KeywordProcessor {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("a", Some("<A#1>"));
        kp
    }

    #[test]
    fn fuzzy_extract_skips_zero_length_match_at_text_start() {
        let kp = processor_with_long_clean_name();
        assert_eq!(kp.extract_keywords("b", Some(0.8)), vec![]);
        assert_eq!(kp.replace_keywords("b", Some(0.8)), "b");
    }

    #[test]
    fn fuzzy_extract_terminates_on_zero_length_match_after_text_start() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let kp = processor_with_long_clean_name();
            let matches: Vec<(usize, usize)> = kp
                .extract_keywords("a b", Some(0.8))
                .iter()
                .map(|m| (m.start, m.end))
                .collect();
            sender.send(matches).unwrap();
        });
        let matches = receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("fuzzy extraction did not finish");
        assert_eq!(matches, vec![(0, 1)]);
    }

    #[test]
    fn fuzzy_matches_start_and_end_on_character_boundaries() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("a", None);
        kp.add_keyword("ab", None);
        let matches = kp.extract_keywords("a😀b", Some(0.5));
        assert_eq!((matches[1].start, matches[1].end), (5, 6));
        assert_eq!(kp.replace_keywords("a😀b", Some(0.5)), "a😀ab");
    }

    #[test]
    fn replace_handles_text_whose_lowercase_is_longer() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("istanbul", Some("IST"));
        assert_eq!(kp.replace_keywords("İstanbul istanbul", None), "İstanbul IST");
    }

    #[test]
    fn non_ascii_keywords_can_be_listed_looked_up_and_removed() {
        let mut kp = KeywordProcessor::new();
        kp.add_keyword("naïve", Some("naive"));
        kp.add_keyword("café", None);
        let all = kp.get_all_keywords();
        assert_eq!(all.get("naïve"), Some(&"naive".to_string()));
        assert_eq!(all.get("café"), Some(&"café".to_string()));
        assert_eq!(kp.get_keyword("NAÏVE"), Some("naive".to_string()));
        assert!(kp.remove_keyword("naïve"));
        assert!(!kp.remove_keyword("naïve"));
        assert_eq!(kp.len(), 1);
    }
}
