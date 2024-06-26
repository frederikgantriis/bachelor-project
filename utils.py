from os import name, system
import os

def extract_words_from_comments(comments):
    extracted_words = {}

    for comment in comments:
        for word in comment:
            if word.text not in extracted_words.keys():
                extracted_words[word.text] = 1
            else:
                extracted_words[word.text] += 1

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


def makedir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
