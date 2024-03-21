from data_parser import Datasets
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.baseline_random import BaselineRandom
from models.binary_naive_bayes import Binary_naive_bayes
from benchmarker import Benchmarker
from constants import TRAIN, TEST

if __name__ == "__main__":
    dataset_train = Datasets(TRAIN)
    dataset_train_2 = Datasets(TRAIN).remove_dots().remove_stop_words()

    dataset_test = Datasets(TEST)

    naive_bayes_1 = NaiveBayes(dataset_train)
    naive_bayes_2 = NaiveBayes(dataset_train_2)
    naive_bayes_2.set_variation_name("no-dots-no-stop-words")

    benchmarker = Benchmarker([naive_bayes_1, naive_bayes_2], dataset_test)

    print(benchmarker.benchmark_models(10, None))
