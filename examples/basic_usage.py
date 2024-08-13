from blitztext import KeywordProcessor


def main():
    keyword_processor = KeywordProcessor(case_sensitive=False)
    keyword_processor.add_keyword('Big Apple', 'New York')
    keyword_processor.add_keyword('Bay Area')
    keyword_processor.add_keyword('Silicon Valley')
    keyword_processor.add_keyword('San Francisco')
    print("add_keyword")

    keywords_found = keyword_processor.extract_keywords('I love Big Apple and Bay Area.')
    print("extract_keywords")
    print(keywords_found)  # ['New York', 'Bay Area']
    print(keywords_found[0].start)  # 7'


if __name__ == '__main__':
    main()
