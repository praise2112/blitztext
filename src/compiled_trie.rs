//! A read-only, cache-friendly copy of the keyword trie (`TrieNode`), used for matching.
//!
//! `KeywordProcessor` keeps the editable `TrieNode` tree as the source of truth (it is what
//! gets pickled) and builds a `CompiledTrie` from it lazily; any add or remove drops it.
//!
//! Layout: a double-array trie over byte classes.
//! - Each distinct byte used by a keyword gets a class 1..n; every other byte is class 0.
//! - State `s` has a slot `[base, check]`. Its child for byte `b` is `t = base[s] + class(b)`,
//!   and that child exists only if `check[t]` equals `s` (low 31 bits). The top bit of
//!   `check[t]` says whether `t` ends a keyword, so an exact walk reads one slot per byte.
//! - Free slots have `check == NONE`; the root is slot 0. Every `base + class` is kept in
//!   bounds, so lookups never need a length check beyond the slice index.
//!
//! Building: states are placed breadth-first. For each state, `FreeSlots::find_base` walks a
//! linked list of free slots for a base where all its children fit. A slot that fails as a
//! candidate too often is dropped from the list, which bounds the search.
//!
//! Per state, `State` also keeps its child edges (for fuzzy search, which must visit every
//! child) and the distance in characters to its deepest keyword (the fuzzy pruning bound).

use crate::fuzzy::TrieView;
use crate::trie::TrieNode;
use std::collections::VecDeque;

const NONE: u32 = u32::MAX;
const KEYWORD_FLAG: u32 = 1 << 31;
const PARENT_MASK: u32 = !KEYWORD_FLAG;

/// Read-only copy of the keyword trie, built for fast matching.
///
/// States live in a double-array: the child of state `s` for byte `b` is
/// `t = base[s] + byte_classes[b]`, valid when `check[t]` names `s` as parent.
/// Fuzzy search also lists each state's children, so those are kept as edge lists.
pub(crate) struct CompiledTrie {
    /// `[base, check]` per slot; `check` holds the parent slot and the keyword flag.
    slots: Vec<[u32; 2]>,
    /// Maps each byte to a class; bytes in no keyword share class 0, which never matches.
    byte_classes: [u32; 256],
    states: Vec<State>,
    edge_bytes: Vec<u8>,
    edge_targets: Vec<u32>,
    keywords: Vec<Box<str>>,
}

#[derive(Clone, Copy)]
struct State {
    first_edge: u32,
    edge_count: u32,
    keyword: u32,
    /// Characters to the deepest keyword at or below this state, or NONE.
    chars_to_deepest_keyword: u32,
}

const EMPTY_STATE: State = State {
    first_edge: 0,
    edge_count: 0,
    keyword: NONE,
    chars_to_deepest_keyword: NONE,
};

/// Free double-array slots in a linked list, for finding bases while building.
#[derive(Default)]
struct FreeSlots {
    next: Vec<u32>,
    previous: Vec<u32>,
    failures: Vec<u8>,
    head: u32,
    tail: u32,
    linked: Vec<bool>,
}

impl FreeSlots {
    /// A slot that failed this many times as a base candidate is no longer tried.
    const MAX_FAILURES: u8 = 16;

    /// Extends `slots` to `length` and links the new slots.
    fn grow(&mut self, slots: &mut Vec<[u32; 2]>, length: usize) {
        if self.linked.is_empty() {
            self.head = NONE;
            self.tail = NONE;
            self.linked.push(false);
            self.next.push(NONE);
            self.previous.push(NONE);
            self.failures.push(0);
        }
        slots.resize(length.max(slots.len()), [0, NONE]);
        for (slot, entry) in slots.iter().enumerate().skip(self.linked.len()) {
            let is_free = entry[1] == NONE;
            self.linked.push(is_free);
            self.next.push(NONE);
            self.previous.push(if is_free { self.tail } else { NONE });
            self.failures.push(0);
            if is_free {
                match self.tail {
                    NONE => self.head = slot as u32,
                    tail => self.next[tail as usize] = slot as u32,
                }
                self.tail = slot as u32;
            }
        }
    }

    fn unlink(&mut self, slot: usize) {
        if !self.linked[slot] {
            return;
        }
        self.linked[slot] = false;
        let (previous, next) = (self.previous[slot], self.next[slot]);
        match previous {
            NONE => self.head = next,
            previous => self.next[previous as usize] = next,
        }
        match next {
            NONE => self.tail = previous,
            next => self.previous[next as usize] = previous,
        }
    }

    /// Finds a base where every `base + class` slot is free. Grows `slots` so that
    /// `base + class` stays in bounds for every class, matching or not.
    fn find_base(&mut self, slots: &mut Vec<[u32; 2]>, classes: &[u32], class_count: u32) -> u32 {
        let Some(&lowest_class) = classes.iter().min() else {
            return 0;
        };
        let mut candidate = self.head;
        loop {
            if candidate == NONE {
                let old_tail = self.tail;
                self.grow(slots, slots.len() + class_count as usize);
                candidate = if old_tail == NONE { self.head } else { self.next[old_tail as usize] };
                continue;
            }
            let slot = candidate as usize;
            if slot > lowest_class as usize {
                let base = (slot - lowest_class as usize) as u32;
                let needed = (base + class_count) as usize + 1;
                if slots.len() < needed {
                    self.grow(slots, needed);
                }
                if classes.iter().all(|&class| slots[(base + class) as usize][1] == NONE) {
                    return base;
                }
                self.failures[slot] += 1;
                if self.failures[slot] >= Self::MAX_FAILURES {
                    self.unlink(slot);
                }
            }
            // Growing may have linked new slots after this one, so read the link now.
            candidate = self.next[slot];
        }
    }
}

impl CompiledTrie {
    pub fn new(root: &TrieNode) -> Self {
        let mut byte_classes = [0u32; 256];
        let mut class_count = 1;
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            for (&key, child) in &node.children {
                let byte = key as u32 as u8;
                if byte_classes[byte as usize] == 0 {
                    byte_classes[byte as usize] = class_count;
                    class_count += 1;
                }
                pending.push(child);
            }
        }

        let mut trie = CompiledTrie {
            slots: vec![[0, NONE]; class_count as usize + 1],
            byte_classes,
            states: Vec::new(),
            edge_bytes: Vec::new(),
            edge_targets: Vec::new(),
            keywords: Vec::new(),
        };
        trie.slots[0][1] = PARENT_MASK;
        let mut free_slots = FreeSlots::default();
        free_slots.grow(&mut trie.slots, class_count as usize + 1);
        let mut order = Vec::new();
        let mut queue = VecDeque::from([(root, 0u32)]);
        while let Some((node, slot)) = queue.pop_front() {
            order.push(slot);
            let classes: Vec<u32> = node
                .children
                .keys()
                .map(|&key| byte_classes[key as u32 as u8 as usize])
                .collect();
            let base = free_slots.find_base(&mut trie.slots, &classes, class_count);
            trie.slots[slot as usize][0] = base;

            if trie.states.len() <= slot as usize {
                trie.states.resize(slot as usize + 1, EMPTY_STATE);
            }
            let state = &mut trie.states[slot as usize];
            state.first_edge = trie.edge_bytes.len() as u32;
            state.edge_count = classes.len() as u32;
            if let Some(keyword) = &node.keyword {
                state.keyword = trie.keywords.len() as u32;
                state.chars_to_deepest_keyword = 0;
                trie.keywords.push(keyword.as_str().into());
            }

            for ((&key, child), class) in node.children.iter().zip(classes) {
                let child_slot = base + class;
                let keyword_flag = if child.keyword.is_some() { KEYWORD_FLAG } else { 0 };
                trie.slots[child_slot as usize][1] = slot | keyword_flag;
                free_slots.unlink(child_slot as usize);
                trie.edge_bytes.push(key as u32 as u8);
                trie.edge_targets.push(child_slot);
                queue.push_back((child, child_slot));
            }
        }
        trie.states.resize(trie.slots.len(), EMPTY_STATE);

        // Children come after their parent in breadth-first order, so walk it backwards.
        for &slot in order.iter().rev() {
            for edge in trie.edges(slot) {
                let child = trie.states[trie.edge_targets[edge] as usize];
                if child.chars_to_deepest_keyword == NONE {
                    continue;
                }
                // UTF-8 continuation bytes do not start a new character.
                let starts_char = trie.edge_bytes[edge] & 0xC0 != 0x80;
                let chars = child.chars_to_deepest_keyword + starts_char as u32;
                let parent = &mut trie.states[slot as usize];
                if parent.chars_to_deepest_keyword == NONE || parent.chars_to_deepest_keyword < chars {
                    parent.chars_to_deepest_keyword = chars;
                }
            }
        }
        trie
    }

    /// Approximate heap and inline size in bytes.
    pub fn memory_usage(&self) -> usize {
        size_of::<Self>()
            + self.slots.capacity() * size_of::<[u32; 2]>()
            + self.states.capacity() * size_of::<State>()
            + self.edge_bytes.capacity()
            + self.edge_targets.capacity() * size_of::<u32>()
            + self.keywords.iter().map(|k| size_of::<Box<str>>() + k.len()).sum::<usize>()
    }

    fn edges(&self, slot: u32) -> std::ops::Range<usize> {
        let state = self.states[slot as usize];
        state.first_edge as usize..(state.first_edge + state.edge_count) as usize
    }

    #[inline]
    fn child(&self, slot: u32, byte: u8) -> Option<u32> {
        let class = self.byte_classes[byte as usize];
        if class == 0 {
            return None;
        }
        let child = self.slots[slot as usize][0] + class;
        (self.slots[child as usize][1] & PARENT_MASK == slot).then_some(child)
    }

    fn keyword(&self, slot: u32) -> Option<&str> {
        let keyword = self.states[slot as usize].keyword;
        (keyword != NONE).then(|| &*self.keywords[keyword as usize])
    }
}

impl<'t> TrieView<'t> for &'t CompiledTrie {
    type Node = u32;

    fn root(&self) -> Self::Node {
        0
    }

    fn node_id(&self, node: Self::Node) -> usize {
        node as usize
    }

    fn clean_name(&self, node: Self::Node) -> Option<&'t str> {
        self.keyword(node)
    }

    fn byte_child(&self, node: Self::Node, byte: u8) -> Option<Self::Node> {
        self.child(node, byte)
    }

    fn for_each_byte_child(&self, node: Self::Node, mut visit: impl FnMut(u8, Self::Node)) {
        for edge in self.edges(node) {
            visit(self.edge_bytes[edge], self.edge_targets[edge]);
        }
    }

    fn chars_to_deepest_keyword(&self, node: Self::Node) -> Option<usize> {
        let chars = self.states[node as usize].chars_to_deepest_keyword;
        (chars != NONE).then_some(chars as usize)
    }

    fn longest_prefix(&self, text: &[u8]) -> Option<(&'t str, usize)> {
        let mut slot = 0;
        let mut found = None;
        for (i, &byte) in text.iter().enumerate() {
            let class = self.byte_classes[byte as usize];
            if class == 0 {
                break;
            }
            // Inlined `child()`: the same check word also carries the keyword flag.
            let child = self.slots[slot as usize][0] + class;
            let check = self.slots[child as usize][1];
            if check & PARENT_MASK != slot {
                break;
            }
            slot = child;
            if check & KEYWORD_FLAG != 0 {
                found = Some((slot, i + 1));
            }
        }
        found.map(|(slot, length)| (self.keyword(slot).unwrap(), length))
    }

}
