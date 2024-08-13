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
}
