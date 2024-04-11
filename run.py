import numpy as np
from constants import TEST, TRAIN
from data_builder import *
from data_parser import Datasets
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.baseline_majority import BaselineMajority
from benchmarker import Benchmarker

def add_baseline_models(to_be_benchmarked, train_datasets, test_datasets):
    # Add BaselineRandom and BaselineMajority models to the list, both trained on the first training dataset
    to_be_benchmarked += [
        (BaselineRandom(train_datasets[0]), test_datasets[0]),
        (BaselineMajority(train_datasets[0]), test_datasets[0]),
    ]
    return to_be_benchmarked

def add_standard_naive_bayes_models(to_be_benchmarked, train_datasets, test_datasets):
    # For each training dataset, create a NaiveBayes model and pair it with the corresponding test dataset
    to_be_benchmarked += [
        (
            NaiveBayes(train_datasets[i], variation_name=variation_names[i]),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]
    return to_be_benchmarked

def add_naive_bayes_models_with_k_factors(to_be_benchmarked, train_datasets, test_datasets):
    # Add more NaiveBayes models to the list, this time with varying k_factor values
    to_be_benchmarked += [
        (
            NaiveBayes(
                train_datasets[i], variation_name=variation_names[i], k_factor=j
            ),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
        for j in np.arange(0, 1, 0.1)
    ]
    return to_be_benchmarked

def add_logistic_regression_models(to_be_benchmarked, train_datasets, test_datasets):
    # Add LogisticRegression models to the list
    to_be_benchmarked += [
        (LogisticRegression(train_datasets[i], variation_name=variation_names[i]), test_datasets[i])
        for i in range(len(train_datasets))
    ]
    return to_be_benchmarked

if __name__ == "__main__":

    # Get the names of all variations
    variation_names = get_variations()
    # Get all training datasets
    train_datasets = get_train_datasets()
    # Get all testing datasets
    test_datasets = get_test_datasets()


    # Initialize a list to hold all models to be benchmarked
    # For each training dataset, create a NaiveBayes model and pair it with the corresponding test dataset
    to_be_benchmarked = add_standard_naive_bayes_models([], train_datasets, test_datasets)

    # Add more NaiveBayes models to the list, this time with varying k_factor values
    to_be_benchmarked += add_naive_bayes_models_with_k_factors(to_be_benchmarked, train_datasets, test_datasets)

    # Add LogisticRegression models to the list
    to_be_benchmarked += add_logistic_regression_models(to_be_benchmarked, train_datasets, test_datasets)

    # Add BaselineRandom and BaselineMajority models to the list, both trained on the first training dataset
    to_be_benchmarked += add_baseline_models(to_be_benchmarked, train_datasets, test_datasets)

    # Create a Benchmarker object with the list of models to be benchmarked
    benchmarker = Benchmarker(to_be_benchmarked, 10)

    # Print the results of benchmarking the models
    print(benchmarker.benchmark_models())

    benchmarker.create_all_charts()