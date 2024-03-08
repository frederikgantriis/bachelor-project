from os import name, system
from datasets import DatasetDict
from sanitizer import Sanitizer


def extract_sentences_from_label(dataset: DatasetDict, label: str):
    extracted_sentences = []

    for i in range(len(dataset["text"])):
        if dataset["label"][i] == label:
            extracted_sentences.append(dataset["text"][i])

    return extracted_sentences


def extract_words_from_label(dataset: DatasetDict, label: str):
    extracted_words = {}
    sentences = extract_sentences_from_label(dataset, label)

    for s in sentences:
        s = Sanitizer(s).sanitize_simple()
        for word in s:
            if word not in extracted_words:
                extracted_words[word] = 1
            else:
                extracted_words[word] += 1

    return extracted_words


def get_max_value_key(d: dict):
    max_value = max(d.values())

    for key in d.keys():
        if d[key] == max_value:
            return key


def flatten(matrix: list) -> list:
    """flattens a matrix into a list

    Args:
        lst (list): 2d list

    Returns:
        list: 1d list
    """
    return [item for row in matrix for item in row]


def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
