use blitztext::KeywordProcessor;

fn main() {
    let mut kp = KeywordProcessor::new();
    kp.add_keyword("rust", None);
    kp.add_keyword("programming", Some("coding"));
    kp.add_keyword("computer science", Some("CS"));

    let text = "I love rust programming and computer science.";
    let result = kp.replace_keywords(text, None);
    println!("Original text: {}", text);
    println!("Processed text: {}", result);

    let keywords = kp.extract_keywords(text, None);
    println!("Extracted keywords:");
    for keyword in keywords {
        println!("- {} ({}..{})", keyword.keyword, keyword.start, keyword.end);
    }
}
