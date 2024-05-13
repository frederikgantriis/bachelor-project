import itertools
from constants import TEST, TRAIN
from data_parser import Datasets
from multiprocessing import Pool


def get_train_datasets():
    return get_all_dataset_combinations(TRAIN)


def get_test_datasets():
    return get_all_dataset_combinations(TEST)


def apply_attributes(args):
    combo, dataset_type = args
    dataset = Datasets(dataset_type)
    for attr in combo:
        dataset = getattr(dataset, attr)()
    return dataset


def get_all_dataset_combinations(dataset_type):
    methods = Datasets(dataset_type).get_all_attributes()
    combinations = [Datasets(dataset_type)]

    with Pool() as pool:
        for i in range(len(methods)):
            combos = list(itertools.permutations(methods, i + 1))
            pool.processes = len(combos)
            results = pool.map(
                apply_attributes, [(combo, dataset_type) for combo in combos]
            )
            combinations.extend(results)

    return combinations


def get_variations():
    methods = Datasets(TRAIN).get_all_attributes()
    combinations = [""]

    for i in range(len(methods)):
        combinations += [
            "_".join(combo) for combo in itertools.permutations(methods, i + 1)
        ]

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


def get_best_naive_bayes_train():
    return [
        Datasets(TRAIN)
        .lowercase()
        .remove_dots()
        .lemmatize()
        .remove_stop_words()
        .remove_duplicates(),
        Datasets(TRAIN)
        .remove_dots()
        .lowercase()
        .remove_duplicates()
        .lemmatize()
        .remove_stop_words(),
        Datasets(TRAIN)
        .remove_dots()
        .remove_stop_words()
        .lowercase()
        .lemmatize()
        .remove_duplicates(),
        Datasets(TRAIN)
        .remove_stop_words()
        .lowercase()
        .remove_dots()
        .lemmatize()
        .remove_duplicates(),
        Datasets(TRAIN).remove_dots(
        ).remove_stop_words().lowercase().lemmatize(),
        Datasets(TRAIN).remove_dots().lowercase(
        ).remove_stop_words().lemmatize(),
        Datasets(TRAIN).remove_stop_words(
        ).remove_dots().lowercase().lemmatize(),
        Datasets(TRAIN)
        .remove_stop_words()
        .remove_dots()
        .remove_duplicates()
        .lowercase()
        .lemmatize(),
    ]


def get_best_naive_bayes_test():
    return [
        Datasets(TEST)
        .lowercase()
        .remove_dots()
        .lemmatize()
        .remove_stop_words()
        .remove_duplicates(),
        Datasets(TEST)
        .remove_dots()
        .lowercase()
        .remove_duplicates()
        .lemmatize()
        .remove_stop_words(),
        Datasets(TEST)
        .remove_dots()
        .remove_stop_words()
        .lowercase()
        .lemmatize()
        .remove_duplicates(),
        Datasets(TEST)
        .remove_stop_words()
        .lowercase()
        .remove_dots()
        .lemmatize()
        .remove_duplicates(),
        Datasets(TEST).remove_dots(
        ).remove_stop_words().lowercase().lemmatize(),
        Datasets(TEST).remove_dots().lowercase(
        ).remove_stop_words().lemmatize(),
        Datasets(TEST).remove_stop_words(
        ).remove_dots().lowercase().lemmatize(),
        Datasets(TEST)
        .remove_stop_words()
        .remove_dots()
        .remove_duplicates()
        .lowercase()
        .lemmatize(),
    ]


def get_best_naive_bayes_variations():
    return [
        "lowercase_remove_dots_lemmatize_remove_stop_words_remove_duplicates",
        "remove_dots_lowercase_remove_duplicates_lemmatize_remove_stop_words",
        "remove_dots_remove_stop_words_lowercase_lemmatize_remove_duplicates",
        "remove_stop_words_lowercase_remove_dots_lemmatize_remove_duplicates",
        "remove_dots_remove_stop_words_lowercase_lemmatize",
        "remove_dots_lowercase_remove_stop_words_lemmatize",
        "remove_stop_words_remove_dots_lowercase_lemmatize",
        "remove_stop_words_remove_dots_remove_duplicates_lowercase_lemmatize",
    ]
