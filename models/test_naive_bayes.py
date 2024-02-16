from data_parser import get_test_dataset, get_train_dataset
from naive_bayes import NaiveBayes, count_words


def test_naive_bayes():
    data = NaiveBayes(get_train_dataset())
    isinstance(data.test(get_test_dataset()["text"]), list)


def test_count_words():
    words = {"a": 1, "b": 2, "c": 3}
    vocabulary = ["a", "b", "c", "d"]
    assert count_words(words, vocabulary) == 10
