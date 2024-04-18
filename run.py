import numpy as np
from data_builder import *
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.baseline_majority import BaselineMajority
from benchmarker import Benchmarker


def add_baseline_models(train_datasets, test_datasets, variation_names):
    # Add BaselineRandom and BaselineMajority models to the list, both trained on the first training dataset
    return [
        (BaselineRandom(train_datasets[0]), test_datasets[0]),
        (BaselineMajority(train_datasets[0]), test_datasets[0]),
    ]


def add_standard_naive_bayes_models(train_datasets, test_datasets, variation_names):
    # For each training dataset, create a NaiveBayes model and pair it with the corresponding test dataset
    return [
        (
            NaiveBayes(train_datasets[i], variation_name=variation_names[i]),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]


def add_naive_bayes_models_with_k_factors(
    train_datasets, test_datasets, variation_names
):
    # Add more NaiveBayes models to the list, this time with varying k_factor values

    return [
        (
            NaiveBayes(
                train_datasets[i], variation_name=variation_names[i], k_factor=k_factor
            ),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
        for k_factor in np.arange(0.1, 1, 0.1)
    ]


def add_logistic_regression_models(train_datasets, test_datasets, variation_names):
    # Add LogisticRegression models to the list
    return [
        (
            LogisticRegression(
                train_datasets[i], variation_name=variation_names[i]),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]


if __name__ == "__main__":

    # Get the names of all variations
    variation_names = get_variations()
    # Get all training datasets
    train_datasets = get_train_datasets()
    # Get all testing datasets
    test_datasets = get_test_datasets()

    # train_datasets = [Datasets(TRAIN)]
    # test_datasets = [Datasets(TEST)]
    # variation_names = [""]

    # Initialize a list to hold all models to be benchmarked
    # For each training dataset, create a NaiveBayes model and pair it with the corresponding test dataset
    to_be_benchmarked = add_standard_naive_bayes_models(
        train_datasets, test_datasets, variation_names
    )

    # Add more NaiveBayes models to the list, this time with varying k_factor values
    to_be_benchmarked += add_naive_bayes_models_with_k_factors(
        train_datasets, test_datasets, variation_names
    )

    # Add LogisticRegression models to the list
    # to_be_benchmarked += add_logistic_regression_models(to_be_benchmarked, train_datasets, test_datasets, variation_names)

    # Add BaselineRandom and BaselineMajority models to the list, both trained on the first training dataset
    to_be_benchmarked += add_baseline_models(
        train_datasets, test_datasets, variation_names
    )

    # Create a Benchmarker object with the list of models to be benchmarked
    benchmarker = Benchmarker(to_be_benchmarked, 10)

    # Print the results of benchmarking the models
    benchmarker.benchmark_models()

    # benchmarker.create_all_charts()
