from models.utils import sanitize, extract_words_from_label
from datasets import DatasetDict
import math
import pickle


class Model:
    def __init__(self) -> None:
        pass


def train(dataset: DatasetDict):
    n_documents = len(dataset["text"])
    classes = set(dataset["label"])
    logprior = {}

    vocabulary = [item for row in [
        sanitize(comment) for comment in dataset["text"]] for item in row]

    for c in classes:
        n_classes = dataset["label"].count(c)
        logprior[c] = math.log10(n_classes / n_documents)
        # bigdoc[c] = extract_sentences_from_label(dataset, c)
        bigwords = extract_words_from_label(dataset, c)
        loglikelihood = {}

        for word in vocabulary:
            count = bigwords[word] if word in bigwords else 0

            loglikelihood[(word, c)] = math.log10(
                (count + 1) / (count_words(bigwords, vocabulary) - count))

    return logprior, loglikelihood, vocabulary


def test(testdoc, logprior, loglikelihood, classes, vocabulary):
    sum = {}
    for c in classes:
        sum[c] = logprior[c]
        for word in testdoc:
            if word in vocabulary:
                sum[c] += loglikelihood[(word, c)]

    print(sum[max(zip(sum[c].values(), sum[c].keys()))[1]])


def count_words(words: dict, vocabulary: list):
    sum = 0
    for word in vocabulary:
        if word in words:
            sum += (words[word] + 1)
        else:
            sum += 1

    return sum


def store_train_data(logprior: dict, loglikelihood: dict, vocabulary: list):
    """Stores the trained data in the computed_data/ folder (fails if the folder doesn't exist)

    Args:
        logprior (dict): the logprior dict returned after training the model (the train() method)
        loglikelihood (dict): the loglikelihood dict returned after training the model (the train() method)
        vocabulary (list): list of all words in all classes (i.e all words in the dataset)
    """
    with open('computed_data/nb_ll.pkl', 'wb') as f:
        pickle.dump(loglikelihood, f)
    with open('computed_data/nb_lp.pkl', 'wb') as f:
        pickle.dump(logprior, f)
    with open('computed_data/nb_v.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)


def load_train_data() -> tuple[dict, dict, list]:
    """loads all naive-bayes train data from computed_data/ folder

    Returns:
        tuple[dict, dict, list]: loglikelihodd, logprior, vocabulary
    """
    with open('computed_data/nb_ll.pkl', 'wb') as f:
        loglikelihood = pickle.load(f)
    with open('computed_data/nb_lp.pkl', 'wb') as f:
        logprior = pickle.load(f)
    with open('computed_data/nb_v.pkl', 'wb') as f:
        vocabulary = pickle.load(f)
    return loglikelihood, logprior, vocabulary


def main(dataset: DatasetDict):
    ll, lp, v = train(dataset)
    print(ll)
