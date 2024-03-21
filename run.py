from data_parser import Datasets
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.baseline_random import BaselineRandom
from analytics.benchmarker import Benchmarker
from constants import TRAIN, TEST

if __name__ == "__main__":
    dataset_train = Datasets(TRAIN)
    dataset_train.remove_dots()

    dataset_test = Datasets(TEST)
    dataset_test.remove_dots().lemmatize().lowercase().remove_stop_words()
    nb1 = NaiveBayes(dataset_train)
    nb1.test(dataset_test.to_list())
    lr = LogisticRegression(dataset_train)
    lr.test(dataset_test.to_list())

    benchmarker = Benchmarker(
        [
            NaiveBayes(dataset_train),
            LogisticRegression(dataset_train),
            BaselineRandom(dataset_train),
        ],
        dataset_test,
    )

    benchmarker.create_all_charts(10)
