import re
from datasets import DatasetDict

labels = ["OFF", "NOT"]


def get_labels():
    return labels


def sanitize(line):
    return re.findall(r'[a-øA-Ø0-9-]+|[^a-zæøåA-ZÆØÅ0-9\s]+', line)


print(sanitize("Ikea-aber"))


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
        s = sanitize(s)
        for word in s:
            if word not in extracted_words:
                extracted_words[word] = 1
            else:
                extracted_words[word] += 1

    return extracted_words


def sanitize_all_lower(line):
    words = sanitize(line)

    for word in words:
        word = word.toLower()

    return words
