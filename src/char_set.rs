use serde::{Deserialize, Serialize, Serializer, Deserializer};

/// The size of the bitmap used to represent the character set.
const BITMAP_SIZE: usize = (char::MAX as usize) / 64 + 1;

/// A fast set implementation for characters.
///
/// `CharSet` uses a bitmap to represent which characters are in the set,
/// allowing for constant-time insertion and membership tests.
///
/// # Examples
///
/// ```
/// use blitztext::CharSet;
///
/// let mut set = CharSet::new();
/// set.insert('a');
/// assert!(set.contains('a'));
/// assert!(!set.contains('b'));
/// ```
#[derive(Debug)]
pub struct CharSet {
    /// The bitmap representing the character set.
    /// Each bit represents a character, where 1 indicates presence in the set.
    pub bitmap: [u64; BITMAP_SIZE],
}

impl Serialize for CharSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert the bitmap to a Vec<u64> for serialization
        let vec: Vec<u64> = self.bitmap.to_vec();
        vec.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CharSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<u64> = Vec::deserialize(deserializer)?;
        if vec.len() != BITMAP_SIZE {
            return Err(serde::de::Error::custom(
                format!("Expected {} elements, got {}", BITMAP_SIZE, vec.len())
            ));
        }
        let mut bitmap = [0; BITMAP_SIZE];
        bitmap.copy_from_slice(&vec);
        Ok(CharSet { bitmap })
    }
}

impl CharSet {
    /// Creates a new, empty CharSet.
    ///
    /// # Returns
    ///
    /// A new CharSet instance with no characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::CharSet;
    ///
    /// let set = CharSet::new();
    /// assert!(!set.contains('a'));
    /// ```
    pub fn new() -> Self {
        CharSet {
            bitmap: [0; BITMAP_SIZE],
        }
    }

    /// Inserts a character into the set.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to insert.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::CharSet;
    ///
    /// let mut set = CharSet::new();
    /// set.insert('a');
    /// assert!(set.contains('a'));
    /// ```
    pub fn insert(&mut self, c: char) {
        let index = c as usize;
        self.bitmap[index / 64] |= 1 << (index % 64);
    }

    /// Extends the set with characters from an iterator.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator yielding characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::CharSet;
    ///
    /// let mut set = CharSet::new();
    /// set.extend('a'..='z');
    /// assert!(set.contains('a'));
    /// assert!(set.contains('z'));
    /// ```
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = char>,
    {
        for c in iter {
            self.insert(c);
        }
    }

    /// Checks if the set contains a specific character.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to check for.
    ///
    /// # Returns
    ///
    /// `true` if the character is in the set, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use blitztext::CharSet;
    ///
    /// let mut set = CharSet::new();
    /// set.insert('a');
    /// assert!(set.contains('a'));
    /// assert!(!set.contains('b'));
    /// ```
    pub fn contains(&self, c: char) -> bool {
        let index = c as usize;
        (self.bitmap[index / 64] & (1 << (index % 64))) != 0
    }
}
