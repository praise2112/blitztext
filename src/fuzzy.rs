//! Fuzzy prefix search over a keyword trie, shared by the editable and the compiled trie.
//!
//! At each word start, `best_fuzzy_prefix` finds the keyword `k` and text span `w` (at least
//! one character) maximising `1 - edits(w, k) / max(|w|, |k|)` in characters, at or above the
//! threshold. Ties go to a span ending in a letter or digit, then the longer span, then the
//! smaller clean name.
//!
//! Search: depth-first over states (trie node, text characters consumed, edits so far).
//! Moves are exact (free), substitution, insertion (skip a text character) and deletion (skip a
//! keyword character), each one edit. Edit moves are taken only while
//! `1 - edits / (span + 1) >= threshold`; at the end of the text only deletions remain.
//!
//! The result does not depend on visit order, so two prunings leave it unchanged:
//! - Bound: a match below a state costs at least its edits and is normalised by at most
//!   `max(span, keyword chars) + chars to the deepest keyword below`.
//! - Memo: a branching state seen again at the same (node, span) with no fewer edits can
//!   only reproduce candidates found from the first visit.
//!
//! The exact child is pushed last, so it is explored first.

use rustc_hash::FxHashMap;
use std::cmp::Reverse;

/// Read access to a keyword trie, implemented by the editable and the compiled trie.
pub(crate) trait TrieView<'t>: Copy {
    type Node: Copy;

    fn root(&self) -> Self::Node;
    /// A key that identifies the node within this trie.
    fn node_id(&self, node: Self::Node) -> usize;
    /// The clean name stored at the node, if the node ends a keyword.
    fn clean_name(&self, node: Self::Node) -> Option<&'t str>;
    fn byte_child(&self, node: Self::Node, byte: u8) -> Option<Self::Node>;
    fn for_each_byte_child(&self, node: Self::Node, visit: impl FnMut(u8, Self::Node));
    /// An upper bound on the characters from this node to the deepest keyword below it,
    /// or None when no keyword is at or below it.
    fn chars_to_deepest_keyword(&self, node: Self::Node) -> Option<usize>;
    /// Returns the longest keyword that `text` starts with, and its length in bytes.
    fn longest_prefix(&self, text: &[u8]) -> Option<(&'t str, usize)>;

    fn char_child(&self, node: Self::Node, c: char) -> Option<Self::Node> {
        if c.is_ascii() {
            return self.byte_child(node, c as u8);
        }
        let mut buffer = [0; 4];
        c.encode_utf8(&mut buffer)
            .bytes()
            .try_fold(node, |node, byte| self.byte_child(node, byte))
    }

    /// Visits each child one whole character below `node`, decoding multi-byte paths.
    fn for_each_char_child(&self, node: Self::Node, mut visit: impl FnMut(char, Self::Node)) {
        fn walk<'t, T: TrieView<'t>>(
            trie: &T,
            node: T::Node,
            bytes: &mut ([u8; 4], usize),
            visit: &mut impl FnMut(char, T::Node),
        ) {
            trie.for_each_byte_child(node, |byte, child| {
                if bytes.1 == 0 && byte.is_ascii() {
                    visit(byte as char, child);
                    return;
                }
                bytes.0[bytes.1] = byte;
                bytes.1 += 1;
                match std::str::from_utf8(&bytes.0[..bytes.1]) {
                    Ok(text) => visit(text.chars().next().unwrap(), child),
                    Err(error) if error.error_len().is_none() => walk(trie, child, bytes, visit),
                    Err(_) => {}
                }
                bytes.1 -= 1;
            });
        }
        walk(self, node, &mut ([0; 4], 0), &mut visit);
    }
}

/// Buffers reused across the fuzzy searches of one text.
pub(crate) struct FuzzyScratch<N> {
    /// Search states: (node, text characters consumed, cost, keyword characters).
    stack: Vec<(N, usize, usize, usize)>,
    lowest_expanded_cost: FxHashMap<(usize, usize), usize>,
    /// Characters of the text being searched, with the byte offset where each ends.
    chars: Vec<(char, usize)>,
}

impl<N> Default for FuzzyScratch<N> {
    fn default() -> Self {
        FuzzyScratch {
            stack: Vec::new(),
            lowest_expanded_cost: FxHashMap::default(),
            chars: Vec::new(),
        }
    }
}

struct Candidate<'t> {
    clean_name: &'t str,
    chars: usize,
    bytes: usize,
    similarity: f32,
    ends_alphanumeric: bool,
}

impl Candidate<'_> {
    /// Higher similarity wins, then a span ending in a letter or digit, then the longer span,
    /// then the smaller clean name.
    fn beats(&self, other: &Candidate) -> bool {
        let rank = |c: &Candidate| (c.similarity, c.ends_alphanumeric, c.chars);
        (rank(self), Reverse(self.clean_name)) > (rank(other), Reverse(other.clean_name))
    }
}

/// Finds the keyword most similar to a prefix of `text`: `1 - edits / max(span, keyword length)`
/// in characters. An edit is allowed only while `1 - edits / (span + 1)` stays at `threshold`.
/// Returns (clean name, matched length in bytes, similarity).
pub(crate) fn best_fuzzy_prefix<'t, T: TrieView<'t>>(
    trie: T,
    text: &str,
    threshold: f32,
    scratch: &mut FuzzyScratch<T::Node>,
) -> (Option<&'t str>, usize, f32) {
    let FuzzyScratch { stack, lowest_expanded_cost, chars } = scratch;
    lowest_expanded_cost.clear();
    chars.clear();
    let mut unread = text.char_indices();
    let mut char_at = |index: usize, chars: &mut Vec<(char, usize)>| {
        while chars.len() <= index {
            let (offset, c) = unread.next()?;
            chars.push((c, offset + c.len_utf8()));
        }
        Some(chars[index].0)
    };

    let mut best: Option<Candidate<'t>> = None;
    // Every match below a state costs at least `cost` and is normalised by at most
    // `max(span, keyword length) + chars to the deepest keyword`.
    let can_beat = |node: T::Node, span: usize, cost: usize, keyword_chars: usize, best: &Option<Candidate>| {
        let Some(remaining) = trie.chars_to_deepest_keyword(node) else {
            return false;
        };
        let longest = span.max(keyword_chars).saturating_add(remaining);
        let best_possible = if cost == 0 { 1.0 } else { 1.0 - (cost as f32 / longest as f32) };
        best_possible >= threshold && best.as_ref().is_none_or(|best| best_possible >= best.similarity)
    };

    stack.clear();
    stack.push((trie.root(), 0, 0, 0));
    while let Some((node, span, cost, keyword_chars)) = stack.pop() {
        if !can_beat(node, span, cost, keyword_chars, &best) {
            continue;
        }
        if let (Some(clean_name), true) = (trie.clean_name(node), span > 0) {
            let similarity = 1.0 - (cost as f32 / span.max(keyword_chars) as f32);
            let candidate = Candidate {
                clean_name,
                chars: span,
                bytes: chars[span - 1].1,
                similarity,
                ends_alphanumeric: chars[span - 1].0.is_alphanumeric(),
            };
            if similarity >= threshold && best.as_ref().is_none_or(|best| candidate.beats(best)) {
                best = Some(candidate);
            }
        }

        // None once the text is used up; only deletions remain possible then.
        let current_char = char_at(span, chars);
        let can_branch = 1.0 - (cost as f32 / (span + 1) as f32) >= threshold;
        if can_branch {
            // A state reached again at no lower cost yields only matches that are no better.
            let lowest = lowest_expanded_cost.entry((trie.node_id(node), span)).or_insert(usize::MAX);
            if *lowest <= cost {
                continue;
            }
            *lowest = cost;
            // Insertion: skip the current text character
            if current_char.is_some() {
                stack.push((node, span + 1, cost + 1, keyword_chars));
            }
            trie.for_each_char_child(node, |c, child| {
                if current_char.is_some_and(|current| c != current) {
                    // Substitution
                    stack.push((child, span + 1, cost + 1, keyword_chars + 1));
                }
                // Deletion: skip the keyword character
                stack.push((child, span, cost + 1, keyword_chars + 1));
            });
        }
        // Pushed last so the exact path is searched first and finds strong matches early.
        if let Some(child) = current_char.and_then(|current| trie.char_child(node, current)) {
            stack.push((child, span + 1, cost, keyword_chars + 1));
        }
    }
    match best {
        Some(best) => (Some(best.clean_name), best.bytes, best.similarity),
        None => (None, 0, 0.0),
    }
}
