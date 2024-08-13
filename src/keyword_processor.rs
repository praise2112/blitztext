use crate::char_set::CharSet;
use crate::trie::TrieNode;
use crate::utils::{is_whitespace, round};
use rayon::prelude::*;
use std::cmp::{max, min};
use std::collections::HashMap;

use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// A struct for handling keyword matching, fuzzy matching, and replacement in text.
#[derive(Serialize, Deserialize)]
pub struct KeywordProcessor {
    /// Set of characters that are considered part of a word.
    non_word_boundaries: CharSet,
    /// The root of the keyword trie.
    root: TrieNode,
    /// Whether the keyword matching should be case-sensitive.
    case_sensitive: bool,
    /// Whether to allow overlapping matches.
    allow_overlaps: bool,
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
        let mut non_word_boundaries = CharSet::new();
        non_word_boundaries.extend('a'..='z');
        non_word_boundaries.extend('A'..='Z');
        non_word_boundaries.extend('0'..='9');
        non_word_boundaries.insert('_');
        KeywordProcessor {
            non_word_boundaries,
            root: TrieNode::new(),
            case_sensitive,
            allow_overlaps,
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

        let mut current_node = &mut self.root;
        // we use bytes as char instead of char because extract_keywords uses bytes
        for byte in keyword.bytes() {
            current_node = current_node
                .children
                .entry(byte as char)
                .or_insert_with(|| Box::new(TrieNode::new()));
        }
        current_node.keyword = Some(clean_name);
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

        Self::remove_keyword_helper(&keyword, &mut self.root)
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
    fn remove_keyword_helper(keyword: &str, node: &mut TrieNode) -> bool {
        if keyword.is_empty() {
            if node.keyword.is_some() {
                node.keyword = None;
                true
            } else {
                false
            }
        } else {
            let (first_char, rest) = keyword.split_at(1);
            if let Some(child) = node.children.get_mut(&first_char.chars().next().unwrap()) {
                let removed = Self::remove_keyword_helper(rest, child);
                if removed && child.children.is_empty() && child.keyword.is_none() {
                    node.children.remove(&first_char.chars().next().unwrap());
                }
                removed
            } else {
                false
            }
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
        self.collect_keywords(&self.root, String::new(), &mut keywords);
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
        prefix: String,
        keywords: &mut HashMap<String, String>,
    ) {
        if let Some(clean_name) = &node.keyword {
            keywords.insert(prefix.clone(), clean_name.clone());
        }
        for (ch, child) in &node.children {
            let mut new_prefix = prefix.clone();
            new_prefix.push(*ch);
            self.collect_keywords(child, new_prefix, keywords);
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
        self.non_word_boundaries = CharSet::new();
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
        let keyword = if self.case_sensitive {
            keyword.to_string()
        } else {
            keyword.to_lowercase()
        };

        let mut current_node = &self.root;
        for ch in keyword.chars() {
            if let Some(next_node) = current_node.children.get(&ch) {
                current_node = next_node;
            } else {
                return None;
            }
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

        // Initialize vector to store found matches
        let mut matches = Vec::new();

        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        // working with bytes is faster than strings
        let text_bytes = text.as_bytes();

        // Iterate through each character in the text
        let mut i = 0;
        while i < text_bytes.len() {
            // Check if we're at a word boundary
            // This is true if we're at the start of the text or the previous character is not a word character
            if i == 0 || !self.is_non_word_boundary(text_bytes[i - 1] as char) {
                // Choose between fuzzy and exact matching based on threshold
                // let (found, length, similarity) = self.find_keyword(&text_bytes[i..]);
                let (found, length, similarity) = if threshold < 1.0 {
                    self.find_keyword_fuzzy(&text_bytes[i..], threshold)
                } else {
                    self.find_keyword(&text_bytes[i..])
                };

                // If a keyword is found
                if let Some(keyword) = found {
                    if threshold == 1.0 {
                        // For exact matching, keep the word boundary check
                        // For ensuring that we only match complete words, not partial words or substrings within larger words.
                        // This is true if we're at the end of the text or the next character is not a word character
                        if i + length == text_bytes.len()
                            || !self.is_non_word_boundary(text_bytes[i + length] as char)
                        {
                            matches.push(KeywordMatch {
                                keyword,
                                similarity: round(similarity, 2),
                                start: if is_whitespace(text_bytes[i]) {
                                    i + 1
                                } else {
                                    i
                                },
                                end: i + length,
                            });
                            // Move the index past this keyword
                            i += length;
                            continue;
                        }
                    } else {
                        // For fuzzy matching, add the match with lenient boundary check
                        // on of next 3 characters is a non-word boundary
                        let end = if is_whitespace(text_bytes[i + length - 1]) {
                            i + length - 1
                        } else {
                            i + length
                        };
                        let is_boundary_in_next_x_chars = (end..min(end + 3, text.len()))
                            .any(|j| !self.is_non_word_boundary(text_bytes[j] as char));
                        if is_boundary_in_next_x_chars || end == text_bytes.len() {
                            matches.push(KeywordMatch {
                                keyword,
                                similarity: round(similarity, 2),
                                start: if is_whitespace(text_bytes[i]) {
                                    i + 1
                                } else {
                                    i
                                },
                                end,
                            });
                            // Move the index past this keyword
                            i += length;
                            continue;
                        }
                    }
                }
            }
            // If no keyword found or not at a word boundary, move to the next character
            i += 1;
        }

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

        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        let text_bytes = text.as_bytes();
        let processor = Arc::new(self);

        let chunk_size = (text_bytes.len() / rayon::current_num_threads()).max(1000);

        (0..text_bytes.len())
            .into_par_iter()
            .step_by(chunk_size)
            .flat_map(|start| {
                let end = (start + chunk_size).min(text_bytes.len());
                let processor = Arc::clone(&processor);

                let mut local_matches = Vec::new();
                let mut i = start;
                while i < end {
                    if i == 0 || !processor.is_non_word_boundary(text_bytes[i - 1] as char) {
                        let (found, length, similarity) = if threshold < 1.0 {
                            processor.find_keyword_fuzzy(&text_bytes[i..], threshold)
                        } else {
                            processor.find_keyword(&text_bytes[i..])
                        };

                        if let Some(keyword) = found {
                            if threshold == 1.0 {
                                if i + length == text_bytes.len()
                                    || !processor
                                        .is_non_word_boundary(text_bytes[i + length] as char)
                                {
                                    local_matches.push(KeywordMatch {
                                        keyword,
                                        similarity: round(similarity, 2),
                                        start: if is_whitespace(text_bytes[i]) {
                                            i + 1
                                        } else {
                                            i
                                        },
                                        end: i + length,
                                    });
                                    i += length;
                                    continue;
                                }
                            } else {
                                let end = if is_whitespace(text_bytes[i + length - 1]) {
                                    i + length - 1
                                } else {
                                    i + length
                                };
                                let is_boundary_in_next_x_chars =
                                    (end..min(end + 3, text_bytes.len())).any(|j| {
                                        !processor.is_non_word_boundary(text_bytes[j] as char)
                                    });
                                if is_boundary_in_next_x_chars || end == text_bytes.len() {
                                    local_matches.push(KeywordMatch {
                                        keyword,
                                        similarity: round(similarity, 2),
                                        start: if is_whitespace(text_bytes[i]) {
                                            i + 1
                                        } else {
                                            i
                                        },
                                        end,
                                    });
                                    i += length;
                                    continue;
                                }
                            }
                        }
                    }
                    i += 1;
                }
                local_matches
            })
            .collect()
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

    /// Helper method to find a keyword starting from a given position in the text.
    ///
    /// This method traverses the trie to find an exact match for a keyword in the text.
    ///
    /// # Arguments
    ///
    /// * `text_bytes` - The text to search in, as a byte slice.
    ///
    /// # Returns
    ///
    /// A tuple containing an Option with the found keyword (if any), the length of the match, and the similarity (1.0 for exact matches).
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(m), where m is the length of the matched keyword.
    fn find_keyword(&self, text_bytes: &[u8]) -> (Option<&str>, usize, f32) {
        // covert text_bytes to string
        let mut current_node = &self.root;
        let mut last_keyword_node = None; // Last node representing a keyword
        let mut last_keyword_pos = 0; // Position of the last keyword found

        // for each character in the text starting from the current position
        // check if there is a matching child node in the trie
        for (i, &byte) in text_bytes.iter().enumerate() {
            // Move to the next node in the trie
            // println!("byte: {:?}, char: {:?}, is_byte_valid_char: {:?}", byte, byte as char, is_valid_char(&[byte]));
            if let Some(next_node) = &current_node.children.get(&(byte as char)) {
                // if let Some(next_node) = &current_node.children[byte as usize] {
                current_node = next_node;
                if current_node.keyword.is_some() {
                    // If this node represents a keyword, remember it
                    last_keyword_node = Some(current_node);
                    last_keyword_pos = i;
                }
            } else {
                break;
            }
        }
        // println!("last_keyword_node: {:?}, last_keyword_pos: {:?}", last_keyword_node, last_keyword_pos);
        if let Some(node) = last_keyword_node {
            (node.keyword.as_deref(), last_keyword_pos + 1, 1.0)
        } else {
            (None, 0, 0.0)
        }
    }

    /// Helper method to find a keyword starting from a given position in the text with fuzzy matching.
    ///
    /// This method uses a depth-first search through the trie to find approximate matches,
    /// allowing for insertions, deletions, and substitutions up to the specified threshold.
    ///
    /// # Arguments
    ///
    /// * `text_bytes` - The text to search in, as a byte slice.
    /// * `threshold` - The minimum similarity required for a match (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// A tuple containing an Option with the found keyword (if any), the length of the match, and the similarity.
    ///
    /// # Complexity
    ///
    /// * Time complexity: O(n * m), where n is the length of the text and m is the average length of keywords.
    ///   In practice, the threshold and early termination optimizations make it much faster for most inputs.
    fn find_keyword_fuzzy(&self, text_bytes: &[u8], threshold: f32) -> (Option<&str>, usize, f32) {
        // Initialize variables to keep track of the best match found
        // let mut best_match: Option<String> = None;
        let mut best_match: Option<&str> = None;
        let mut best_match_length = 0;
        let mut best_similarity = 0.0;

        // Stack for depth-first search through the trie
        // Each element is (node, depth in text, cumulative cost)
        let mut stack = vec![(&self.root, 0, 0)];

        // Depth-first search through the trie
        while let Some((node, depth, cost)) = stack.pop() {
            // Check if this node represents a keyword
            if let Some(keyword) = &node.keyword {
                let keyword_len = keyword.len();
                // Use the maximum of current depth and keyword length for normalization
                let max_length = max(depth, keyword_len);
                // Calculate similarity as 1 - (cost / max_length)
                let similarity = 1.0 - (cost as f32 / max_length as f32);
                // println!("similarity: {:?}, depth: {:?}, cost: {:?}, s>=t: {:?}, best_similarity: {:?}, best_match: {:?}", similarity, depth, cost, similarity >= threshold, best_similarity, best_match);

                // Update the best match if the similarity is above the threshold
                if similarity >= threshold && similarity > best_similarity {
                    best_match = Some(keyword);
                    // best_match = Some(keyword.clone());
                    best_match_length = depth;
                    best_similarity = similarity;
                }
            }

            // Stop exploring this path if we've reached the end of the text, or it's word boundary
            if depth >= text_bytes.len() {
                // if depth >= text_bytes.len() || (depth >5 && !self.is_non_word_boundary(text_bytes[depth-5] as char)) {
                continue;
            }

            // Get the current byte(char) in the text
            let current_char = text_bytes[depth] as char;

            // Explore exact matches (no additional cost)
            if let Some(child) = &node.children.get(&current_char) {
                stack.push((child, depth + 1, cost));
            }

            // Explore fuzzy matches
            // Calculate current similarity to determine if we should continue exploring
            let current_similarity = 1.0 - (cost as f32 / (depth + 1) as f32);
            if current_similarity >= threshold {
                // Insertion: skip the current character in the text (cost + 1)
                stack.push((node, depth + 1, cost + 1));

                // Deletion and Substitution
                for (ch, child) in &node.children {
                    if *ch != current_char {
                        // Substitution: try all children with increased cost (cost + 1)
                        stack.push((child, depth + 1, cost + 1));
                    }
                    // Deletion: skip the current node in the trie (cost + 1)
                    stack.push((child, depth, cost + 1));
                }
            }
        }
        // println!("best_match: {:?}, best_match_length: {:?}, best_similarity: {:?}", best_match, best_match_length, best_similarity);
        (best_match, best_match_length, best_similarity)
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
        let matches = self.extract_keywords(text, threshold);
        let mut result = text.to_string();

        // Sort matches by start position in descending order
        let mut sorted_matches = matches;
        sorted_matches.sort_by(|a, b| b.start.cmp(&a.start));

        for keyword_match in sorted_matches {
            let start = keyword_match.start;
            let end = keyword_match.end;
            result.replace_range(start..end, &keyword_match.keyword);
        }

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
        // Check if the character is in the set of non-word boundaries
        self.non_word_boundaries.contains(ch)
        // self.non_word_boundaries.contains(&ch)
    }

    /// Calculates the approximate memory usage of the trie in bytes.
    ///
    /// This method provides an estimate of the memory used by the trie structure,
    /// including all nodes and the strings stored in them. It does not include
    /// the memory used by the KeywordProcessor struct itself.
    ///
    /// # Returns
    ///
    /// The estimated memory usage in bytes.
    pub fn calculate_memory_usage(&self) -> usize {
        let trie_memory = self.calculate_node_memory(&self.root);

        // Add size of KeywordProcessor struct itself
        trie_memory + size_of::<KeywordProcessor>()
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
        println!("m  {:?}", (char::MAX as usize) / 64 + 1);

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
        println!("KKKK: {:?}", matches);
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
        println!("{:?}", keywords);

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
        println!("res: {:?}", result);
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
                    similarity: 0.78,
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
        println!("{:?}", keywords);

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
        println!("{:?}", keywords);
        assert_eq!(keywords.len(), 0);
    }
}
