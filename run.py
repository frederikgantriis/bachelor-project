from data_parser import Datasets
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.baseline_random import BaselineRandom
from models.binary_naive_bayes import Binary_naive_bayes
from benchmarker import Benchmarker
from constants import TRAIN, TEST

if __name__ == "__main__":
    dataset_train = Datasets(TRAIN)
    dataset_train.remove_dots()

    dataset_test = Datasets(TEST)
    dataset_test.remove_dots()
    nb1 = NaiveBayes(dataset_train)
    nb1.test(dataset_test.to_list())
    lr = LogisticRegression(dataset_train)
    lr.test(dataset_test.to_list())

    benchmarker = Benchmarker(
        [
            NaiveBayes(dataset_train),
            LogisticRegression(dataset_train),
            BaselineRandom(dataset_train),
            Binary_naive_bayes(dataset_train),
        ],
        dataset_test,
    )

    print(benchmarker.benchmark_models(1, None))
