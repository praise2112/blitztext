import multiprocessing
import random
import re
import string
import time

import matplotlib.pyplot as plt
import spacy
import stringzilla as sz
from blitztext import KeywordProcessor as BlitzTextProcessor
from flashtext import KeywordProcessor
from spacy.matcher import PhraseMatcher

# Constants for benchmark configuration
WORDS_PER_TEXT = 5000
NUM_TEXTS_PARALLEL = 1000
MAX_KEYWORDS = 20001
STEP_SIZE = 1000

# Load spaCy model with only the tokenizer
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])


def get_word_of_length(str_length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(str_length))


def find_keywords(text, keywords):
    return [text.find(kw) for kw in keywords]


def spacy_match(doc, matcher):
    return matcher(doc)


def stringzilla_match(text, keywords):
    return [sz.find(text, kw) for kw in keywords]


def benchmark_keyword_extraction(keywords_length, all_words):
    sample_text_words = random.sample(all_words, WORDS_PER_TEXT)
    sample_text = ' '.join(sample_text_words)
    unique_keywords_sublist = list(set(random.sample(all_words, keywords_length)))

    compiled_re = re.compile('|'.join([r'\b' + keyword + r'\b' for keyword in unique_keywords_sublist]))

    flashtext_processor = KeywordProcessor()
    flashtext_processor.add_keywords_from_list(unique_keywords_sublist)

    blitztext_processor = BlitzTextProcessor()
    for keyword in unique_keywords_sublist:
        blitztext_processor.add_keyword(keyword, None)

    # Prepare Spacy PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in unique_keywords_sublist]
    matcher.add("KeywordList", patterns)

    start = time.time()
    _ = blitztext_processor.extract_keywords(sample_text, None)
    mid1 = time.time()
    _ = compiled_re.findall(sample_text)
    mid2 = time.time()
    _ = flashtext_processor.extract_keywords(sample_text)
    mid3 = time.time()
    _ = find_keywords(sample_text, unique_keywords_sublist)
    mid4 = time.time()
    doc = nlp(sample_text)
    _ = spacy_match(doc, matcher)
    mid5 = time.time()
    _ = stringzilla_match(sample_text, unique_keywords_sublist)
    end = time.time()

    return {
        "keywords": keywords_length,
        "blitztext": mid1 - start,
        "regex": mid2 - mid1,
        "flashtext": mid3 - mid2,
        "python_find": mid4 - mid3,
        "spacy": mid5 - mid4,
        "stringzilla": end - mid5
    }


def benchmark_keyword_extraction_parallel(keywords_length, all_words):
    sample_texts = [' '.join(random.sample(all_words, WORDS_PER_TEXT)) for _ in range(NUM_TEXTS_PARALLEL)]
    unique_keywords_sublist = list(set(random.sample(all_words, keywords_length)))

    compiled_re = re.compile('|'.join([r'\b' + keyword + r'\b' for keyword in unique_keywords_sublist]))

    flashtext_processor = KeywordProcessor()
    flashtext_processor.add_keywords_from_list(unique_keywords_sublist)

    blitztext_processor = BlitzTextProcessor()
    for keyword in unique_keywords_sublist:
        blitztext_processor.add_keyword(keyword, None)

    # Prepare Spacy PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in unique_keywords_sublist]
    matcher.add("KeywordList", patterns)

    start = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.map(flashtext_processor.extract_keywords, sample_texts)
    mid1 = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.map(compiled_re.findall, sample_texts)
    mid2 = time.time()
    _ = blitztext_processor.parallel_extract_keywords_from_texts(sample_texts, None)
    mid3 = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.starmap(find_keywords, [(text, unique_keywords_sublist) for text in sample_texts])
    mid4 = time.time()
    docs = list(nlp.pipe(sample_texts))
    _ = [spacy_match(doc, matcher) for doc in docs]
    mid5 = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.starmap(stringzilla_match, [(text, unique_keywords_sublist) for text in sample_texts])
    end = time.time()

    return {
        "keywords": keywords_length,
        "flashtext": mid1 - start,
        "regex": mid2 - mid1,
        "blitztext": mid3 - mid2,
        "python_find": mid4 - mid3,
        "spacy": mid5 - mid4,
        "stringzilla": end - mid5
    }


def run_benchmark_single_threaded(benchmark_func):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for _ in range(100000)]
    return [benchmark_func(keywords_length, all_words) for keywords_length in range(0, MAX_KEYWORDS, STEP_SIZE)]


def run_benchmark_multi_threaded(benchmark_func):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for _ in range(100000)]
    return [benchmark_func(keywords_length, all_words) for keywords_length in range(0, MAX_KEYWORDS, STEP_SIZE)]


def plot_results(results, filename, title, num_texts=1):
    plt.figure(figsize=(12, 6))

    for method in ["flashtext", "regex", "blitztext", "python_find", "spacy", "stringzilla"]:
        plt.plot([r["keywords"] for r in results], [r[method] for r in results], label=method.capitalize(), marker='o')

    plt.xlabel('Number of Keywords')
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add benchmark details to the plot
    details = f"Benchmark Details:\n"
    details += f"Words per text: {WORDS_PER_TEXT}\n"
    details += f"Number of texts processed: {num_texts}\n"
    details += f"Max keywords: {MAX_KEYWORDS}\n"
    details += f"Step size: {STEP_SIZE}"
    plt.text(0.05, 0.95, details, transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")


def write_results_to_markdown(single_extraction, multi_extraction, filename='benchmark_results.md'):
    with open(filename, 'w') as f:
        f.write("# Benchmark Results\n\n")

        f.write("## Benchmark Configuration\n\n")
        f.write(f"- Words per text: {WORDS_PER_TEXT}\n")
        f.write(f"- Number of texts for parallel: {NUM_TEXTS_PARALLEL}\n")
        f.write(f"- Maximum number of keywords: {MAX_KEYWORDS - 1}\n")
        f.write(f"- Step size for keyword count: {STEP_SIZE}\n\n")

        f.write("## Keyword Extraction (Single-threaded)\n\n")
        f.write("| Keywords | FlashText | Regex | BlitzText | Python find() | Spacy | StringZilla |\n")
        f.write("|----------|-----------|-------|-----------|---------------|-------|-------------|\n")
        for result in single_extraction:
            f.write(
                f"| {result['keywords']} | {result['flashtext']:.5f} | {result['regex']:.5f} | {result['blitztext']:.5f} | {result['python_find']:.5f} | {result['spacy']:.5f} | {result['stringzilla']:.5f} |\n")

        f.write("\n## Keyword Extraction (Multi-threaded)\n\n")
        f.write("| Keywords | FlashText | Regex | BlitzText | Python find() | Spacy | StringZilla |\n")
        f.write("|----------|-----------|-------|-----------|---------------|-------|-------------|\n")
        for result in multi_extraction:
            f.write(
                f"| {result['keywords']} | {result['flashtext']:.5f} | {result['regex']:.5f} | {result['blitztext']:.5f} | {result['python_find']:.5f} | {result['spacy']:.5f} | {result['stringzilla']:.5f} |\n")

    print(f"Results written to {filename}")


if __name__ == '__main__':
    print("Running single-threaded extraction benchmark...")
    single_extraction = run_benchmark_single_threaded(benchmark_keyword_extraction)
    print("Running multi-threaded extraction benchmark...")
    multi_extraction = run_benchmark_multi_threaded(benchmark_keyword_extraction_parallel)
    # multi_extraction = []

    plot_results(single_extraction, 'benchmark_results_single.png', 'Single-threaded Performance Comparison')
    plot_results(multi_extraction, 'benchmark_results_multi.png', 'Multi-threaded Performance Comparison',
                 NUM_TEXTS_PARALLEL)
    write_results_to_markdown(single_extraction, multi_extraction)
