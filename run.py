import numpy as np
from data_builder import *
from models.baseline_random import BaselineRandom
from models.n_gram_logistic_regression import NGramLogisticRegression
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.baseline_majority import BaselineMajority
from benchmarker import Benchmarker
from models.svm import SVM


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
            LogisticRegression(train_datasets[i], variation_name=variation_names[i]),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]


def add_ngram_logistic_regression_models(
    train_datasets, test_datasets, variation_names
):
    # Add LogisticRegression models to the list
    return [
        (
            NGramLogisticRegression(
                train_datasets[i], variation_name=variation_names[i]
            ),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]


def add_svm_models(train_datasets, test_datasets, variation_names):
    return [
        (
            SVM(train_datasets[i], variation_name=variation_names[i]),
            test_datasets[i],
        )
        for i in range(len(train_datasets))
    ]


if __name__ == "__main__":
    # Load the dataset

    train = [
        Datasets(TRAIN)
        .remove_stop_words()
        .lemmatize()
        .extract_unique_words()
        .remove_dots()
        .lowercase()
    ]
    test = [
        Datasets(TEST)
        .remove_stop_words()
        .lemmatize()
        .extract_unique_words()
        .remove_dots()
        .lowercase()
    ]
    name = ["remove_stop_words_lemmatize_extract_unique_words_remove_dots_lowercase"]
    to_be_benchmarked = add_ngram_logistic_regression_models(train, test, name)
    for _ in range(49):
        # Add ngram logistic regression models to the list
        to_be_benchmarked += add_ngram_logistic_regression_models(train, test, name)

    benchmarker = Benchmarker(to_be_benchmarked, 1)

    # Print the results of benchmarking the models
    benchmarker.benchmark_models()
