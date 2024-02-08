from models.utils import sanitize
from datasets import DatasetDict

off_words = {}
not_words = {}

def classify_words(dataset: DatasetDict):

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


    return off_words, not_words


def amount_of_off():
    return off_words.keys() / (off_words.keys() + not_words.keys())

def amount_of_not():
    return not_words.keys() / (off_words.keys() + not_words.keys())
        
