import math
import pickle

import pytest
from blitztext import KeywordProcessor
from blitztext import PyKeywordMatch as KeywordMatch


@pytest.fixture
def processor():
    kp = KeywordProcessor()
    kp.add_keyword("rust", "Rust Lang")
    return kp


@pytest.mark.parametrize("threshold", [-0.1, 1.5, math.nan])
def test_threshold_outside_zero_to_one_raises_value_error(processor, threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        processor.extract_keywords("rust", threshold)
    with pytest.raises(ValueError):
        processor.parallel_replace_keywords_from_texts(["rust"], threshold)


def test_corrupt_pickle_raises_value_error(processor):
    with pytest.raises(ValueError):
        processor.__setstate__(b"\x01\x02")


def test_processor_and_matches_survive_pickling(processor):
    restored = pickle.loads(pickle.dumps(processor))
    [match] = restored.extract_keywords("I love rust")
    assert (match.keyword, match.start, match.end) == ("Rust Lang", 7, 11)
    assert repr(pickle.loads(pickle.dumps(match))) == repr(match)


def test_match_keyword_can_be_reassigned():
    match = KeywordMatch("a", 1.0, 0, 1)
    match.keyword = "b"
    assert match.keyword == "b"
