use std::time::Instant;
//
// use plotters::prelude::*;
use blitztext::KeywordProcessor;
use rand::seq::IndexedRandom;
use rand::Rng;
use regex::Regex;

fn get_word_of_length(str_length: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..str_length)
        .map(|_| rng.gen_range(b'a'..=b'z') as char)
        .collect()
}

fn benchmark_keyword_extraction() -> Vec<(usize, f64, f64)> {
    let mut rng = rand::thread_rng();
    let all_words: Vec<String> = (0..100000)
        .map(|_| get_word_of_length(rng.gen_range(3..=8)))
        .collect();

    let mut results = Vec::new();
    println!("Count  | FlashText | Regex    ");
    println!("-------------------------------");

    for keywords_length in (0..=20000).step_by(1000) {
        let all_words_chosen = all_words
            .choose_multiple(&mut rng, 5000)
            .collect::<Vec<_>>();
        let story = all_words_chosen
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        // let story = all_words_chosen.iter().cloned().collect::<Vec<_>>().join(" ");
        // let story = all_words_chosen.join(" ");
        let unique_keywords_sublist: Vec<String> = all_words
            .choose_multiple(&mut rng, keywords_length)
            .cloned()
            .collect();

        // let regex = regex::RegexSet::new(unique_keywords_sublist.iter().map(|k| format!(r"\b{}\b", k))).unwrap(); //
        // let regex_pattern = unique_keywords_sublist.iter().map(|k| regex::escape(k)).collect::<Vec<_>>().join("|");
        // let regex = Regex::new(&format!(r"\b({})\b", regex_pattern)).unwrap();
        // Corrected regex pattern to match Python implementation
        let regex_pattern = unique_keywords_sublist
            .iter()
            .map(|k| format!(r"\b{}\b", regex::escape(k)))
            .collect::<Vec<_>>()
            .join("|");
        let regex = Regex::new(&regex_pattern).unwrap();

        let mut keyword_processor = KeywordProcessor::new();
        keyword_processor.add_keywords_from_list(
            &unique_keywords_sublist
                .iter()
                .map(|k| (k.as_str(), None))
                .collect::<Vec<_>>(),
        );

        let start = Instant::now();
        // let _ = keyword_processor.extract_keywords(&story, Some(0.9));
        // let _ = keyword_processor.extract_keywords_parallel(&story, None);
        // let _ = keyword_processor.extract_keywords(&story);
        let _ = keyword_processor.extract_keywords(&story, None);
        let mid = Instant::now();
        // let _ = regex.matches(&story); //
        let _ = regex.find_iter(&story).count();
        let end = Instant::now(); //

        let flashtext_time = mid.duration_since(start).as_secs_f64();
        let regex_time = end.duration_since(mid).as_secs_f64(); //

        results.push((keywords_length, flashtext_time, regex_time)); //

        println!(
            "{:<6} | {:.5}   | {:.5}",
            keywords_length, flashtext_time, regex_time
        ); //
    }

    results
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark_keyword_extraction();
    Ok(())
}
