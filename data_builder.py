import itertools
from constants import TEST, TRAIN
from data_parser import Datasets


def get_train_datasets():
    return get_all_dataset_combinations(TRAIN)

def get_test_datasets():
    return get_all_dataset_combinations(TEST)

def get_all_dataset_combinations(dataset_type):
    methods = Datasets(dataset_type).get_all_attributes()
    combinations = [Datasets(dataset_type)]

    for i in range(len(methods)):
        for combo in itertools.permutations(methods, i + 1):
            dataset = Datasets(dataset_type)
            for attr in combo:
                dataset = getattr(dataset, attr)()
            combinations.append(dataset)

    return combinations


def get_variations():
    methods = Datasets(TRAIN).get_all_attributes()
    combinations = [""]

    for i in range(len(methods)):
        combinations += ['_'.join(combo) for combo in itertools.permutations(methods, i + 1)]

    return combinations


def get_logistic_regression_train():
    return [
        Datasets(TRAIN),
        Datasets(TRAIN).remove_dots().lowercase(),
        Datasets(TRAIN).lemmatize(),
        Datasets(TRAIN).remove_dots().remove_stop_words(),
        Datasets(TRAIN).remove_dots().lowercase().remove_stop_words(),
        Datasets(TRAIN).remove_dots().lemmatize().remove_stop_words(),
    ]


def get_logistic_regression_test():
    return [
        Datasets(TEST),
        Datasets(TEST).remove_dots().lowercase(),
        Datasets(TEST).lemmatize(),
        Datasets(TEST).remove_dots().remove_stop_words(),
        Datasets(TEST).remove_dots().lowercase().remove_stop_words(),
        Datasets(TEST).remove_dots().lemmatize().remove_stop_words(),
    ]


def get_logistic_regression_variations():
    return [
        "",
        "remove_dots_lowercase",
        "lemmatize",
        "remove_dots_remove_stop_words",
        "remove_dots_lowercase_remove_stop_words",
        "remove_dots_lemmatize_remove_stop_words",
    ]
