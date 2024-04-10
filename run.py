from constants import TEST, TRAIN
from data_builder import *
from data_parser import Datasets
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.baseline_majority import BaselineMajority
from benchmarker import Benchmarker


if __name__ == "__main__":

    variation_names = get_variations()
    train_datasets = get_train_datasets()
    test_datasets = get_test_datasets()

    # train_datasets = [Datasets(TRAIN)]
    # test_datasets = [Datasets(TEST)]

    # train_datasets = get_logistic_regression_train()
    # test_datasets = get_logistic_regression_test()

    to_be_benchmarked = []

    for i in range(len(train_datasets)):
        nb = NaiveBayes(train_datasets[i], variation_name=variation_names[i])
        to_be_benchmarked.append((nb, test_datasets[i]))

        aknb = NaiveBayes(train_datasets[i], variation_name=variation_names[i], k_factor=0.5)
        to_be_benchmarked.append((aknb, test_datasets[0]))

    baseline_random = BaselineRandom(train_datasets[0])
    to_be_benchmarked.append((baseline_random, test_datasets[0]))

    baseline_majority = BaselineMajority(train_datasets[0])
    to_be_benchmarked.append((baseline_majority, test_datasets[0]))

    benchmarker = Benchmarker(to_be_benchmarked)
    print(benchmarker.benchmark_models(20))



