import utils
from datasets import DatasetDict
import math
import pickle


def train(dataset: DatasetDict):
    n_documents = len(dataset["text"])
    classes = set(dataset["label"])
    logprior = {}
    loglikelihood = {}

    vocabulary = [item for row in [
        sanitize(comment) for comment in dataset["text"]] for item in row]

    for c in classes:
        n_classes = dataset["label"].count(c)
        logprior[c] = math.log10(n_classes / n_documents)
        print("extracting words...")
        bigwords = extract_words_from_label(dataset, c)
        n_words = count_words(bigwords, vocabulary)
        print("words extracted")
        print(n_words)

        for word in vocabulary:
            count = bigwords[word] if word in bigwords else 0

            print("test:", count + 1, n_words - count)
            loglikelihood[(word, c)] = math.log10(
                (count + 1) / (n_words - count))

    return logprior, loglikelihood, vocabulary


def test(testdoc, logprior, loglikelihood, classes, vocabulary):
    sum = {}
    for c in classes:
        sum[c] = logprior[c]
        for word in testdoc:
            try:
                sum[c] += loglikelihood[(word, c)]
            except KeyError:
                continue

    print(sum)


def store_train_data(logprior: dict, loglikelihood: dict, vocabulary: list):

    with open('computed_data/nb_lp.pkl', 'wb') as f:
        pickle.dump(logprior, f)
    with open('computed_data/nb_ll.pkl', 'wb') as f:
        pickle.dump(loglikelihood, f)
    with open('computed_data/nb_v.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)


def load_train_data() -> tuple[dict, dict, list]:
    """loads all naive-bayes train data from computed_data/ folder

    Returns:
        tuple[dict, dict, list]: logprior, loglikelihood, vocabulary
    """
    with open('computed_data/nb_lp.pkl', 'rb') as f:
        logprior = pickle.load(f)
    with open('computed_data/nb_ll.pkl', 'rb') as f:
        loglikelihood = pickle.load(f)
    with open('computed_data/nb_v.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    return logprior, loglikelihood, vocabulary


def main(dataset: DatasetDict):
    # lp, ll, v = train(dataset)
    # store_train_data(lp, ll, v)

    lp, ll, v = load_train_data()
    test(sanitize("hej med dig"),
         lp, ll, set(dataset["label"]), v)
