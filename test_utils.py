import utils


def test_sanitize():
    assert utils.sanitize("Hei, jeg heter Ola Nordmann.") == [
        "Hei",
        ",",
        "jeg",
        "heter",
        "Ola",
        "Nordmann",
        ".",
    ]


def test_extract_sentences_from_label():
    dataset = {
        "text": ["Hei, jeg heter Ola Nordmann.", "Hei, jeg heter Kari Nordmann."],
        "label": ["OFF", "NOT"],
    }
    assert utils.extract_sentences_from_label(dataset, "OFF") == [
        "Hei, jeg heter Ola Nordmann."
    ]


def test_extract_words_from_label():
    dataset = {
        "text": ["Hei, jeg heter Ola Nordmann. Hei", "Hei, jeg heter Kari Nordmann."],
        "label": ["OFF", "NOT"],
    }
    assert utils.extract_words_from_label(dataset, "OFF") == {
        "Hei": 2,
        ",": 1,
        "jeg": 1,
        "heter": 1,
        "Ola": 1,
        "Nordmann": 1,
        ".": 1,
    }


def test_get_max_value_key():
    d = {"Hei": 1, ",": 1, "jeg": 1, "heter": 1, "Ola": 2, "Nordmann": 1, ".": 1}
    assert utils.get_max_value_key(d) == "Ola"


def test_flatten():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert utils.flatten(matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sanitize_all_lower():
    assert utils.sanitize_all_lower("Hei, jeg heter Ola Nordmann.") == [
        "hei",
        ",",
        "jeg",
        "heter",
        "ola",
        "nordmann",
        ".",
    ]


def test_remove_stop_words():
    assert utils.remove_stop_words("hej, jeg hedder ola nordmann.") == [
        "hej",
        ",",
        "hedder",
        "ola",
        "nordmann",
        ".",
    ]
