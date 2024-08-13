use blitztext::KeywordProcessor;

#[test]
fn test_keyword_processor_integration() {
    // Initialize a case-insensitive KeywordProcessor
    let mut processor = KeywordProcessor::new();

    // Test adding keywords
    processor.add_keyword("rust", Some("Rust Programming Language"));
    processor.add_keyword("python", Some("Python Programming Language"));
    processor.add_keyword("go", Some("Go Programming Language"));
    processor.add_keyword("java", Some("Java Programming Language"));

    // Test adding keywords from a list
    let keywords = vec![
        ("c++", Some("C++ Programming Language")),
        ("javascript", Some("JavaScript Programming Language")),
    ];
    processor.add_keywords_from_list(&keywords);

    // Test getting all keywords
    let all_keywords = processor.get_all_keywords();
    assert_eq!(all_keywords.len(), 6);
    assert_eq!(
        all_keywords.get("rust"),
        Some(&"Rust Programming Language".to_string())
    );

    // Test extracting keywords
    let text = "I love programming in Rust and Python. Sometimes I use Go or Java too.";
    let matches = processor.extract_keywords(text, None);
    assert_eq!(matches.len(), 4);
    assert_eq!(matches[0].keyword, "Rust Programming Language");
    assert_eq!(matches[1].keyword, "Python Programming Language");
    assert_eq!(matches[2].keyword, "Go Programming Language");
    assert_eq!(matches[3].keyword, "Java Programming Language");

    // Test replacing keywords
    let replaced_text = processor.replace_keywords(text, None);
    assert_eq!(replaced_text, "I love programming in Rust Programming Language and Python Programming Language. Sometimes I use Go Programming Language or Java Programming Language too.");

    // Test fuzzy matching
    let fuzzy_text = "I love programming in Rast and Pythn. Sometimes I use Jva too.";
    processor.remove_keyword("go");
    let fuzzy_matches = processor.extract_keywords(fuzzy_text, Some(0.8));
    assert_eq!(fuzzy_matches.len(), 3);
    assert!(fuzzy_matches.iter().all(|m| m.similarity >= 0.8));

    // Test case sensitivity
    let mut case_sensitive_processor = KeywordProcessor::with_options(true, false);
    case_sensitive_processor.add_keyword("Rust", Some("Rust Programming Language"));
    case_sensitive_processor.add_keyword("rust", Some("rust (oxidation)"));

    let case_text = "I love Rust, but I don't like rust on my car.";
    let case_matches = case_sensitive_processor.extract_keywords(case_text, None);
    assert_eq!(case_matches.len(), 2);
    assert_eq!(case_matches[0].keyword, "Rust Programming Language");
    assert_eq!(case_matches[1].keyword, "rust (oxidation)");

    // Test removing keywords
    assert!(processor.remove_keyword("python"));
    assert_eq!(processor.len(), 4);

    let removed_matches = processor.extract_keywords(text, None);
    assert_eq!(removed_matches.len(), 2);
    assert!(!removed_matches
        .iter()
        .any(|m| m.keyword == "Python Programming Language"));

    // Test non-word boundaries
    processor.set_non_word_boundaries(&['_']);
    processor.add_keyword("under_score", Some("underscore"));

    let boundary_text = "This_is_an_under_score_test";
    let boundary_matches = processor.extract_keywords(boundary_text, None);
    assert_eq!(boundary_matches.len(), 0);

    // Test parallel extraction
    let texts = vec!["Rust is awesome", "Java is widely used", "C++ is powerful"];
    let parallel_results = processor.parallel_extract_keywords_from_texts(&texts, None);
    assert_eq!(parallel_results.len(), 3);
    assert!(parallel_results[0]
        .iter()
        .any(|m| m.keyword == "Rust Programming Language"));
    assert!(parallel_results[1]
        .iter()
        .any(|m| m.keyword == "Java Programming Language"));
    assert!(parallel_results[2]
        .iter()
        .any(|m| m.keyword == "C++ Programming Language"));

    // Test memory usage
    let memory_usage = processor.calculate_memory_usage();
    assert!(memory_usage > 0);
}
