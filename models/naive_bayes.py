from models.utils import sanitize, extract_sentences_from_label, extract_words_from_label
from datasets import DatasetDict
import pickle
import math

off_words = {}
not_words = {}


def classify_words(dataset: DatasetDict, label: str = None) -> dict:

    comments = dataset["text"]
    labels = dataset["label"]

    for i in range(len(comments)):
        words = sanitize(comments[i])

        for word in words:
            if labels[i] == "OFF":
                if word in off_words.keys():
                    off_words[word] += 1
                else:
                    off_words[word] = 1
            else:
                if word in not_words.keys():
                    not_words[word] += 1
                else:
                    not_words[word] = 1
        if label == "OFF":
            return off_words
        if label == "NOT":
            return not_words


def amount_of_off():
    return off_words.keys() / (off_words.keys() + not_words.keys())


def amount_of_not():
    return not_words.keys() / (off_words.keys() + not_words.keys())


def train(dataset: DatasetDict):
    n_documents = len(dataset["text"])
    classes = set(dataset["label"])
    logprior = {}

    # TODO: please make this readable
    vocabulary = [item for row in [
        sanitize(comment) for comment in dataset["text"]] for item in row]

    print(classes)
    for c in classes:
        n_classes = dataset["label"].count(c)
        logprior[c] = math.log10(n_classes / n_documents)
        # bigdoc[c] = extract_sentences_from_label(dataset, c)
        bigwords = extract_words_from_label(dataset, c)
        loglikelihood = {}

        for word in vocabulary:
            count = bigwords[word] if word in bigwords else 0

            loglikelihood[(word, c)] = math.log10(
                (count + 1) / (count_words(bigwords) - count))

    with open('computed_data/nb_ll.pkl', 'wb') as f:
        pickle.dump(loglikelihood, f)

    with open('computed_data/nb_lp.pkl', 'wb') as f:
        pickle.dump(logprior, f)


def test(testdoc, logprior, loglikelihood, classes, vocabulary):
    sum = {}
    for c in classes:
        sum[c] = logprior[c]
        for word in testdoc:
            if word in vocabulary:
                sum[c] += loglikelihood[(word, c)]

    print(sum[max(zip(sum[c].values(), sum[c].keys()))[1]])


def count_words(words: dict):
    sum = 0
    for word in words.keys():
        sum += (words[word] + 1)

    return sum
