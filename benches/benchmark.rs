use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flashtext::{DefaultKeywordProcessor, UnicodeKeywordProcessor};
use std::time::Duration;

fn add_keywords_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add Keywords");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(format!("{} keywords", size), size, |b, &size| {
            b.iter(|| {
                let mut processor = DefaultKeywordProcessor::new(false);
                for i in 0..size {
                    processor.add_keyword(&format!("keyword{}", i), None);
                }
            });
        });
    }
    group.finish();
}

fn extract_keywords_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Extract Keywords");
    group.measurement_time(Duration::from_secs(10));

    let mut processor = DefaultKeywordProcessor::new(false);
    for i in 0..1000 {
        processor.add_keyword(&format!("keyword{}", i), None);
    }

    let texts = [
        "This is a short text with keyword5 and keyword10.",
        "This is a medium length text with keyword100, keyword200, and keyword300. It also contains some other words.",
        "This is a long text with many keywords: keyword1, keyword10, keyword100, keyword500, keyword999. It's designed to test the performance of keyword extraction with a larger text and multiple matches. The text goes on for a while to simulate a more realistic scenario with more content to process."
    ];

    for text in texts.iter() {
        group.bench_with_input(format!("Text length: {}", text.len()), text, |b, text| {
            b.iter(|| {
                processor.extract_keywords(black_box(text), None);
            });
        });
    }

    group.finish();
}

fn parallel_vs_sequential_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel vs Sequential");
    group.measurement_time(Duration::from_secs(10));

    let mut processor = DefaultKeywordProcessor::new(false);
    for i in 0..1000 {
        processor.add_keyword(&format!("keyword{}", i), None);
    }

    let long_text = "This is a very long text with many keywords: keyword1, keyword10, keyword100, keyword500, keyword999. ".repeat(100);

    group.bench_function("Sequential", |b| {
        b.iter(|| {
            processor.extract_keywords(black_box(&long_text), None);
        });
    });

    group.bench_function("Parallel", |b| {
        b.iter(|| {
            processor.extract_keywords_parallel(black_box(&long_text), None);
        });
    });

    group.finish();
}

fn parallel_extract_keywords_from_texts_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Process Multiple Texts");
    group.measurement_time(Duration::from_secs(10));

    let mut processor = DefaultKeywordProcessor::new(false);
    for i in 0..1000 {
        processor.add_keyword(&format!("keyword{}", i), None);
    }

    let text = "This is a sample text with keyword1, keyword10, keyword100.";
    let texts: Vec<String> = (0..100).map(|_| text.to_string()).collect();

    group.bench_function("Extract 100 texts", |b| {
        b.iter(|| {
            processor.parallel_extract_keywords_from_texts(black_box(&texts), None);
        });
    });

    group.finish();
}

// fn fuzzy_matching_benchmark(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Fuzzy Matching");
//     group.measurement_time(Duration::from_secs(10));
//
//     let mut processor = DefaultKeywordProcessor::new(false);
//     for i in 0..1000 {
//         processor.add_keyword(&format!("keyword{}", i), None);
//     }
//
//     let text = "This is a text with keywor1, keyward10, keywodr100, keyword500, keyward999.";
//
//     for threshold in [0.5, 0.7, 0.9, 1.0].iter() {
//         group.bench_with_input(format!("Threshold: {}", threshold), threshold, |b, &threshold| {
//             b.iter(|| {
//                 processor.extract_keywords(black_box(text), Some(threshold));
//             });
//         });
//     }
//
//     group.finish();
// }

fn fuzzy_matching_vs_exact_matching_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fuzzy vs Exact Matching");
    group.measurement_time(Duration::from_secs(10));

    let mut processor = DefaultKeywordProcessor::new(false);
    for i in 0..1000 {
        processor.add_keyword(&format!("keyword{}", i), None);
    }

    let text = "This is a text with keywor1, keyward10, keywodr100, keyword500, keyward999.";

    group.bench_function("Exact Matching", |b| {
        b.iter(|| {
            processor.extract_keywords(black_box(text), None);
        });
    });

    group.bench_function("Fuzzy Matching", |b| {
        b.iter(|| {
            processor.extract_keywords(black_box(text), Some(0.5));
        });
    });

    group.finish();
}

fn case_sensitivity_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Case Sensitivity");
    group.measurement_time(Duration::from_secs(10));

    let mut case_sensitive = DefaultKeywordProcessor::new(true);
    let mut case_insensitive = DefaultKeywordProcessor::new(false);

    for i in 0..1000 {
        let keyword = format!("Keyword{}", i);
        case_sensitive.add_keyword(&keyword, None);
        case_insensitive.add_keyword(&keyword, None);
    }

    let text = "This is a text with Keyword1, KEYWORD10, keyword100, KeyWord500, kEyWoRd999.";

    group.bench_function("Case Sensitive", |b| {
        b.iter(|| {
            case_sensitive.extract_keywords(black_box(text), None);
        });
    });

    group.bench_function("Case Insensitive", |b| {
        b.iter(|| {
            case_insensitive.extract_keywords(black_box(text), None);
        });
    });

    group.finish();
}

fn unicode_vs_ascii_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Unicode vs ASCII");
    group.measurement_time(Duration::from_secs(10));

    let mut ascii_processor = DefaultKeywordProcessor::new(false);
    let mut unicode_processor = UnicodeKeywordProcessor::new(false);

    for i in 0..1000 {
        ascii_processor.add_keyword(&format!("keyword{}", i), None);
        unicode_processor.add_keyword(&format!("keyword{}", i), None);
    }

    let ascii_text = "This is an ASCII text with keyword1, keyword10, keyword100.";
    let unicode_text = "This is a Unicode text with keyword1, keyword10, keyword100, and some Unicode characters: áéíóú ñ ü ß 你好 こんにちは";

    group.bench_function("ASCII Processor with ASCII text", |b| {
        b.iter(|| {
            ascii_processor.extract_keywords(black_box(ascii_text), None);
        });
    });

    group.bench_function("Unicode Processor with ASCII text", |b| {
        b.iter(|| {
            unicode_processor.extract_keywords(black_box(ascii_text), None);
        });
    });

    group.bench_function("ASCII Processor with Unicode text", |b| {
        b.iter(|| {
            ascii_processor.extract_keywords(black_box(unicode_text), None);
        });
    });

    group.bench_function("Unicode Processor with Unicode text", |b| {
        b.iter(|| {
            unicode_processor.extract_keywords(black_box(unicode_text), None);
        });
    });

    group.finish();
}

fn keyword_removal_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Keyword Removal");
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            format!("Remove from {} keywords", size),
            size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let mut processor = DefaultKeywordProcessor::new(false);
                        for i in 0..size {
                            processor.add_keyword(&format!("keyword{}", i), None);
                        }
                        processor
                    },
                    |mut processor| {
                        for i in 0..size {
                            black_box(processor.remove_keyword(&format!("keyword{}", i)));
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

fn replace_keywords_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Replace Keywords");
    group.measurement_time(Duration::from_secs(10));

    let mut processor = DefaultKeywordProcessor::new(false);
    for i in 0..1000 {
        processor.add_keyword(&format!("keyword{}", i), Some(&format!("replaced{}", i)));
    }

    let texts = [
        "This is a short text with keyword5 and keyword10.",
        "This is a medium length text with keyword100, keyword200, and keyword300. It also contains some other words.",
        "This is a long text with many keywords: keyword1, keyword10, keyword100, keyword500, keyword999. It's designed to test the performance of keyword replacement with a larger text and multiple matches. The text goes on for a while to simulate a more realistic scenario with more content to process."
    ];

    for text in texts.iter() {
        group.bench_with_input(format!("Text length: {}", text.len()), text, |b, text| {
            b.iter(|| {
                processor.replace_keywords(black_box(text), None);
            });
        });
    }

    // Fuzzy replacement benchmark
    let fuzzy_text = "This is a text with keywor1, keyward10, keywodr100, keyword500, keyward999.";
    group.bench_with_input("Fuzzy replacement", fuzzy_text, |b, text| {
        b.iter(|| {
            processor.replace_keywords(black_box(text), Some(0.8));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    // add_keywords_benchmark,
    // extract_keywords_benchmark,
    // parallel_vs_sequential_benchmark,
    // parallel_extract_keywords_from_texts_benchmark,
    // fuzzy_matching_benchmark,
    // fuzzy_matching_vs_exact_matching_benchmark,
    // case_sensitivity_benchmark,
    // unicode_vs_ascii_benchmark
    keyword_removal_benchmark,
    replace_keywords_benchmark
);
criterion_main!(benches);
