from os import name, system

def extract_words_from_comments(comments):
    extracted_words = {}

    for s in comments:
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
