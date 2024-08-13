import multiprocessing
import random
import re
import string
import time

import matplotlib.pyplot as plt
from blitztext import KeywordProcessor as BlitzTextProcessor
from flashtext import KeywordProcessor

# Constants for benchmark configuration
WORDS_PER_TEXT = 5000
NUM_TEXTS_PARALLEL = 1000
MAX_KEYWORDS = 20000
STEP_SIZE = 1000


def get_word_of_length(str_length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(str_length))


def benchmark_keyword_extraction(keywords_length, all_words):
    sample_text_words = random.sample(all_words, WORDS_PER_TEXT)
    sample_text = ' '.join(sample_text_words)
    unique_keywords_sublist = list(set(random.sample(all_words, keywords_length)))

    compiled_re = re.compile('|'.join([r'\b' + keyword + r'\b' for keyword in unique_keywords_sublist]))

    flashtext_processor = KeywordProcessor()
    flashtext_processor.add_keywords_from_list(unique_keywords_sublist)

    blitztext_processor = BlitzTextProcessor()
    for keyword in unique_keywords_sublist:
        blitztext_processor.add_keyword(keyword)

    start = time.time()
    _ = blitztext_processor.extract_keywords(sample_text, None)
    mid1 = time.time()
    _ = compiled_re.findall(sample_text)
    mid2 = time.time()
    _ = flashtext_processor.extract_keywords(sample_text)
    mid3 = time.time()
    return {
        "keywords": keywords_length,
        "blitztext": mid1 - start,
        "regex": mid2 - mid1,
        "flashtext": mid3 - mid2
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

    start = time.time()
    _ = blitztext_processor.parallel_extract_keywords_from_texts(sample_texts, None)
    # with multiprocessing.Pool() as pool:
    #     _ = pool.map(blitztext_processor.extract_keywords, sample_texts)
    mid1 = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.map(compiled_re.findall, sample_texts)
    mid2 = time.time()
    with multiprocessing.Pool() as pool:
        _ = pool.map(flashtext_processor.extract_keywords, sample_texts)
    mid3 = time.time()

    return {
        "keywords": keywords_length,
        "blitztext": mid1 - start,
        "regex": mid2 - mid1,
        "flashtext": mid3 - mid2,
    }


def run_benchmark_single_threaded(benchmark_func):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for _ in range(100000)]
    results = []
    for keywords_length in range(0, MAX_KEYWORDS, STEP_SIZE):
        results.append(benchmark_func(keywords_length, all_words))
    return results


def run_benchmark_multi_threaded(benchmark_func):
    all_words = [get_word_of_length(random.choice([3, 4, 5, 6, 7, 8])) for _ in range(100000)]
    results = []
    for keywords_length in range(0, MAX_KEYWORDS, STEP_SIZE):
        results.append(benchmark_func(keywords_length, all_words))
    return results


def plot_results(results, filename, title, num_texts=1):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Plot all methods on the first subplot
    for method in ["flashtext", "blitztext", "regex", ]:
        ax1.plot([r["keywords"] for r in results], [r[method] for r in results], label=method.capitalize(), marker='o')

    ax1.set_xlabel('Number of Keywords')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot only BlitzText and FlashText on the second subplot
    for method in ["flashtext", "blitztext"]:
        ax2.plot([r["keywords"] for r in results], [r[method] for r in results], label=method.capitalize(), marker='o')

    ax2.set_xlabel('Number of Keywords')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('BlitzText vs FlashText Comparison')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add benchmark details to the plot
    details = f"Benchmark Details:\n"
    details += f"Words per text: {WORDS_PER_TEXT}\n"
    details += f"Number of texts processed: {num_texts}\n"
    details += f"Max keywords: {MAX_KEYWORDS}\n"
    details += f"Step size: {STEP_SIZE}"
    fig.text(0.15, 0.9, details, fontsize=8, verticalalignment='top',
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
        f.write("| Keywords | FlashText | Regex | BlitzText |\n")
        f.write("|----------|-----------|-------|-----------|\n")
        for result in single_extraction:
            f.write(
                f"| {result['keywords']} | {result['flashtext']:.5f} | {result['regex']:.5f} | {result['blitztext']:.5f} |\n")

        f.write("\n## Keyword Extraction (Multi-threaded)\n\n")
        f.write("| Keywords | FlashText | Regex | BlitzText |\n")
        f.write("|----------|-----------|-------|-----------|\n")
        for result in multi_extraction:
            f.write(
                f"| {result['keywords']} | {result['flashtext']:.5f} | {result['regex']:.5f} | {result['blitztext']:.5f} |\n")

    print(f"Results written to {filename}")


if __name__ == '__main__':
    print("Running single-threaded extraction benchmark...")
    single_extraction = run_benchmark_single_threaded(benchmark_keyword_extraction)
    print("Running multi-threaded extraction benchmark...")
    time.sleep(10)
    multi_extraction = run_benchmark_multi_threaded(benchmark_keyword_extraction_parallel)
    # multi_extraction = []

    plot_results(single_extraction, 'benchmark_results_single.png', 'Single-threaded Performance Comparison')
    plot_results(multi_extraction, 'benchmark_results_multi.png', 'Multi-threaded Performance Comparison',
                 NUM_TEXTS_PARALLEL)
    write_results_to_markdown(single_extraction, multi_extraction)
