import re
import spacy


class Sanitizer(object):
    def __init__(self, comment: str):
        self.comment: str = comment

    def sanitize_simple(self):
        return re.findall(r"[a-øA-Ø0-9-]+|[^a-zæøåA-ZÆØÅ0-9\s]+", self.comment)

    def sanitize_all_lower(self):
        return [x.lower() for x in self.sanitize_simple(self.comment)]

    def sanitize_only_words(self, line):
        return re.findall(r"[a-øA-Ø0-9-]+", line)

    def remove_stop_words(self) -> list[str]:
        """removes most common words danish words from a string"""
        nlp = spacy.load("da_core_news_sm")
        return [x.text for x in nlp(self.comment) if not x.is_stop]
