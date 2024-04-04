from constants import TEST, TRAIN
from data_builder import *
from data_parser import Datasets
from models.baseline_random import BaselineRandom
from models.binary_naive_bayes import BinaryNaiveBayes
from models.naive_bayes import NaiveBayes
from models.add_k_naive_bayes import AddKNaiveBayes
from models.logistic_regression import LogisticRegression
from benchmarker import Benchmarker


if __name__ == "__main__":

    # train_datasets = get_train_datasets()
    # test_datasets = get_test_datasets()

    # train_datasets = [Datasets(TRAIN)]
    # test_datasets = [Datasets(TEST)]

    train_datasets = get_logistic_regression_train()
    test_datasets = get_logistic_regression_test()

    variation_names = get_logistic_regression_variations()
    to_be_benchmarked = []

    for i in range(len(train_datasets)):
        # nb = NaiveBayes(train_datasets[i], variation_name=variation_names[i])
        # to_be_benchmarked.append((nb, test_datasets[i]))

        # bnb = BinaryNaiveBayes(train_datasets[i], variation_name=variation_names[i])
        # to_be_benchmarked.append((bnb, test_datasets[i]))

        # aknb = AddKNaiveBayes(train_datasets[i], variation_name=variation_names[i])
        # to_be_benchmarked.append((aknb, test_datasets[i]))

        lr = LogisticRegression(train_datasets[i], variation_name=variation_names[i])
        to_be_benchmarked.append((lr, test_datasets[i]))

    baseline_random = BaselineRandom(train_datasets[0])
    to_be_benchmarked.append((baseline_random, test_datasets[0]))

    baseline_majority = BaselineMajority(train_datasets[0])
    to_be_benchmarked.append((baseline_majority, test_datasets[0]))

    benchmarker = Benchmarker(to_be_benchmarked)
    print(benchmarker.benchmark_models(20))
