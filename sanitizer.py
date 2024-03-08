from copy import deepcopy
import re


class Sanitizer(object):
    def __init__(self, dataset: dict[str, list]):
        self.dataset = dataset

        self.simple_func = lambda x: re.findall(
            r"[a-øA-Ø0-9-]+|[^a-zæøåA-ZÆØÅ0-9\s]+", x)

    def sanitize(self, func) -> dict[str, list]:
        dataset_copy: dict[str, list] = deepcopy(self.dataset)

        for key in dataset_copy.keys():
            dataset_copy[key] = [func(word) for word in self.dataset[key]]
        return dataset_copy

    def simple(self):
        return self.sanitize(self.simple_func)

    def sanitize_all_lower(self):
        return [x.lower() for x in self.sanitize_simple(self.sentences)]

    # def sanitize_only_words(self, line):
    #     return re.findall(r"[a-øA-Ø0-9-]+", line)

    # def remove_stop_words(self) -> list[str]:
    #     """removes most common words danish words from a string"""
    #     nlp = spacy.load("da_core_news_sm")
    #     return [x.text for x in nlp(self.sentences) if not x.is_stop]


# def sync_disk(method: str, result: list[str]):
#     if type(result) != list or type(result[0]) != str:
#         raise ValueError("Sanitized data must be a list of strings")

#     try:
#         disk_data = SanitizedDataset(method, []).load_from_disk()
#         print("Successfully loaded from disk!")
#     except FileNotFoundError:
#         print("FileNotFound: Saving to disk before loading...")
#         SanitizedDataset(method, result).save_to_disk()
#         disk_data = SanitizedDataset(method, []).load_from_disk()
#         print("Successfully loaded from disk!")
#     print(disk_data)
