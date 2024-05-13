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
            LogisticRegression(
                train_datasets[i], variation_name=variation_names[i]),
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
    # # remove dots lowercase remove stop words lemmatize
    # train_data2 = Datasets("train").remove_dots(
    # ).lowercase().remove_stop_words().lemmatize()

    # test_data2 = Datasets("test").remove_dots(
    # ).lowercase().remove_stop_words().lemmatize()

    # For each training dataset, create a NaiveBayes model and pair it with the corresponding test dataset
    to_be_benchmarked = add_standard_naive_bayes_models(
        train_datasets, test_datasets, variation_names
    )

    svm = SVM(train_data)
    svm.train()

    # Add LogisticRegression models to the list
    to_be_benchmarked += add_logistic_regression_models(
        train_datasets, test_datasets, variation_names
    )

    # Add BaselineRandom and BaselineMajority models to the list, both trained on the first training dataset
    to_be_benchmarked += add_baseline_models(
        train_datasets, test_datasets, variation_names
    )

    # Add ngram logistic regression models to the list
    to_be_benchmarked += add_ngram_logistic_regression_models(
        train_datasets, test_datasets, variation_names
    )

    # Add SVM models to the list
    to_be_benchmarked += add_svm_models(train_datasets, test_datasets, variation_names)

    # Create a Benchmarker object with the list of models to be benchmarked
    benchmarker = Benchmarker(to_be_benchmarked, 10)

    # Print the results of benchmarking the models
    benchmarker.benchmark_models()
