from typing import Dict, List, Optional, Tuple

class KeywordMatch:
    """
    Represents a matched keyword in the text.

    Attributes:
        keyword (str): The matched keyword or its clean name.
        similarity (float): The similarity score of the match (1.0 for exact matches).
        start (int): The starting index of the match in the text.
        end (int): The ending index of the match in the text.
    """
    keyword: str
    similarity: float
    start: int
    end: int

    def __repr__(self) -> str:
        """
        Returns a string representation of the KeywordMatch object.
        """
        ...

class KeywordProcessor:
    """
    A processor for efficient keyword matching and replacement in text.

    Based on AhoCorasick for fast keyword operations,
    including fuzzy matching capabilities.
    """

    def __init__(self, case_sensitive: Optional[bool] = None, allow_overlaps: Optional[bool] = None) -> None:
        """
        Initialize a new KeywordProcessor.

        Args:
            case_sensitive: If True, matching will be case-sensitive. Default is False.
            allow_overlaps: If True, allows overlapping matches. Default is True.
        """
        ...

    def add_keyword(self, keyword: str, clean_name: Optional[str] = None) -> None:
        """
        Add a keyword to the processor.

        Args:
            keyword: The keyword to add.
            clean_name: An optional 'clean' version of the keyword to return when found.
        """
        ...

    def remove_keyword(self, keyword: str) -> bool:
        """
        Remove a keyword from the processor.

        Args:
            keyword: The keyword to remove.

        Returns:
            True if the keyword was found and removed, False otherwise.
        """
        ...

    def extract_keywords(self, text: str, threshold: Optional[float] = None) -> List[KeywordMatch]:
        """
        Extract keywords from the given text.

        Args:
            text: The text to search for keywords.
            threshold: Threshold for fuzzy matching (0.0 to 1.0). Default is 1.0 (exact matches only).

        Returns:
            A list of KeywordMatch objects representing the found keywords.
        """
        ...

    def parallel_extract_keywords_from_texts(self, texts: List[str], threshold: Optional[float] = None) -> List[List[KeywordMatch]]:
        """
        Extract keywords from multiple texts in parallel.

        Args:
            texts: A list of texts to search for keywords.
            threshold: Threshold for fuzzy matching (0.0 to 1.0). Default is 1.0 (exact matches only).

        Returns:
            A list of lists of KeywordMatch objects, one list per input text.
        """
        ...

    def replace_keywords(self, text: str, threshold: Optional[float] = None) -> str:
        """
        Replace keywords in the given text with their clean names.

        Args:
            text: The text in which to replace keywords.
            threshold: Threshold for fuzzy matching (0.0 to 1.0). Default is 1.0 (exact matches only).

        Returns:
            The text with keywords replaced by their clean names.
        """
        ...

    def parallel_replace_keywords_from_texts(self, texts: List[str], threshold: Optional[float] = None) -> List[str]:
        """
        Replace keywords in multiple texts in parallel.

        Args:
            texts: A list of texts in which to replace keywords.
            threshold: Threshold for fuzzy matching (0.0 to 1.0). Default is 1.0 (exact matches only).

        Returns:
            A list of strings with keywords replaced by their clean names.
        """
        ...

    def get_all_keywords(self) -> Dict[str, str]:
        """
        Get all keywords and their clean names.

        Returns:
            A dictionary where keys are keywords and values are their clean names.
        """
        ...

    def set_non_word_boundaries(self, boundaries: List[str]) -> None:
        """
        Set the characters that are considered part of a word.

        Args:
            boundaries: A list of characters to be set as non-word boundaries.
        """
        ...

    def add_non_word_boundary(self, boundary: str) -> None:
        """
        Add a single character as a non-word boundary.

        Args:
            boundary: The character to be added as a non-word boundary.
        """
        ...

    def __len__(self) -> int:
        """
        Get the number of keywords in the processor.

        Returns:
            The number of keywords.
        """
        ...

    def __repr__(self) -> str:
        """
        Get a string representation of the KeywordProcessor.

        Returns:
            A string representation including the number of keywords.
        """
        ...