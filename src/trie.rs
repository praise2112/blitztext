use crate::fuzzy::TrieView;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// A node in the keyword trie.
///
/// This struct represents a single node in the trie data structure used by the KeywordProcessor.
/// Each node can have up to N children, where N is specified as a const generic parameter.
/// This allows for optimization based on the expected character set (e.g., ASCII or Unicode).
///
/// # Type Parameters
///
/// * `N`: The number of possible children for each node. Defaults to 128 for ASCII compatibility.
///
#[derive(Debug)]
#[derive(Serialize, Deserialize)]
pub(crate) struct TrieNode {
    /// Child nodes, indexed by characters.
    pub children: FxHashMap<char, Box<TrieNode>>,

    /// The keyword stored at this node, if any.
    pub keyword: Option<String>,
}

impl TrieNode {
    /// Creates a new, empty TrieNode.
    ///
    /// # Returns
    ///
    /// A new TrieNode instance with no children and no keyword.
    pub fn new() -> Self {
        TrieNode {
            children: FxHashMap::default(),
            // children: HashMap::new(),
            keyword: None,
        }
    }

    pub fn count_nodes(&self) -> usize {
        let mut count = 0;
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            count += 1;
            pending.extend(node.children.values().map(|child| &**child));
        }
        count
    }
}

impl<'t> TrieView<'t> for &'t TrieNode {
    type Node = &'t TrieNode;

    fn root(&self) -> Self::Node {
        self
    }

    fn node_id(&self, node: Self::Node) -> usize {
        node as *const TrieNode as usize
    }

    fn clean_name(&self, node: Self::Node) -> Option<&'t str> {
        node.keyword.as_deref()
    }

    fn byte_child(&self, node: Self::Node, byte: u8) -> Option<Self::Node> {
        node.children.get(&(byte as char)).map(|child| &**child)
    }

    fn for_each_byte_child(&self, node: Self::Node, mut visit: impl FnMut(u8, Self::Node)) {
        for (&key, child) in &node.children {
            visit(key as u32 as u8, child);
        }
    }

    fn chars_to_deepest_keyword(&self, _node: Self::Node) -> Option<usize> {
        // Every leaf ends a keyword, but the depth is not tracked here.
        Some(usize::MAX)
    }

    fn longest_prefix(&self, text: &[u8]) -> Option<(&'t str, usize)> {
        let mut current_node = *self;
        let mut found = None;
        for (i, &byte) in text.iter().enumerate() {
            match current_node.children.get(&(byte as char)) {
                Some(next_node) => current_node = next_node,
                None => break,
            }
            if let Some(keyword) = &current_node.keyword {
                found = Some((keyword.as_str(), i + 1));
            }
        }
        found
    }
}
