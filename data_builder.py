from constants import TEST, TRAIN
from data_parser import Datasets


def get_train_datasets():
    return [
        Datasets(TRAIN),
        Datasets(TRAIN).remove_dots(),
        Datasets(TRAIN).remove_stop_words(),
        Datasets(TRAIN).lowercase(),
        Datasets(TRAIN).lemmatize(),
        Datasets(TRAIN).remove_dots().remove_stop_words(),
        Datasets(TRAIN).remove_stop_words().remove_dots(),
        Datasets(TRAIN).remove_dots().lowercase(),
        Datasets(TRAIN).lowercase().remove_dots(),
        Datasets(TRAIN).remove_dots().lemmatize(),
        Datasets(TRAIN).lemmatize().remove_dots(),
        Datasets(TRAIN).remove_stop_words().lowercase(),
        Datasets(TRAIN).lowercase().remove_stop_words(),
        Datasets(TRAIN).remove_stop_words().lemmatize(),
        Datasets(TRAIN).lemmatize().remove_stop_words(),
        Datasets(TRAIN).remove_dots().remove_stop_words().lowercase(),
        Datasets(TRAIN).remove_dots().lowercase().remove_stop_words(),
        Datasets(TRAIN).remove_stop_words().remove_dots().lowercase(),
        Datasets(TRAIN).remove_dots().remove_stop_words().lemmatize(),
        Datasets(TRAIN).remove_dots().lemmatize().remove_stop_words(),
        Datasets(TRAIN).remove_stop_words().remove_dots().lemmatize(),
        Datasets(TRAIN).remove_stop_words().lemmatize().remove_dots(),
        Datasets(TRAIN).lemmatize().remove_stop_words().remove_dots(),
        Datasets(TRAIN).remove_dots().remove_stop_words().lowercase().lemmatize(),
        Datasets(TRAIN).remove_dots().remove_stop_words().lemmatize().lowercase(),
        Datasets(TRAIN).remove_dots().lowercase().remove_stop_words().lemmatize(),
        Datasets(TRAIN).remove_stop_words().remove_dots().lowercase().lemmatize(),
        Datasets(TRAIN).remove_stop_words().lowercase().remove_dots().lemmatize(),
        Datasets(TRAIN).lowercase().remove_stop_words().remove_dots().lemmatize(),
        Datasets(TRAIN).remove_dots().remove_stop_words().lowercase().lemmatize()
    ]

def get_variations():
    # Variation names matching datasets in get_train_datasets and in the same order
    return [
        "",
        "remove_dots",
        "remove_stop_words",
        "lowercase",
        "lemmatize",
        "remove_dots_remove_stop_words",
        "remove_stop_words_remove_dots",
        "remove_dots_lowercase",
        "lowercase_remove_dots",
        "remove_dots_lemmatize",
        "lemmatize_remove_dots",
        "remove_stop_words_lowercase",
        "lowercase_remove_stop_words",
        "remove_stop_words_lemmatize",
        "lemmatize_remove_stop_words",
        "remove_dots_remove_stop_words_lowercase",
        "remove_dots_lowercase_remove_stop_words",
        "remove_stop_words_remove_dots_lowercase",
        "remove_dots_remove_stop_words_lemmatize",
        "remove_dots_lemmatize_remove_stop_words",
        "remove_stop_words_remove_dots_lemmatize",
        "remove_stop_words_lemmatize_remove_dots",
        "lemmatize_remove_stop_words_remove_dots",
        "remove_dots_remove_stop_words_lowercase_lemmatize",
        "remove_dots_remove_stop_words_lemmatize_lowercase",
        "remove_dots_lowercase_remove_stop_words_lemmatize",
        "remove_stop_words_remove_dots_lowercase_lemmatize",
        "remove_stop_words_lowercase_remove_dots_lemmatize",
        "lowercase_remove_stop_words_remove_dots_lemmatize",
        "remove_dots_remove_stop_words_lowercase_lemmatize"
    ]

def get_test_datasets():
    # Same order as in get_train_datasets
    return [
        Datasets(TEST),
        Datasets(TEST).remove_dots(),
        Datasets(TEST).remove_stop_words(),
        Datasets(TEST).lowercase(),
        Datasets(TEST).lemmatize(),
        Datasets(TEST).remove_dots().remove_stop_words(),
        Datasets(TEST).remove_stop_words().remove_dots(),
        Datasets(TEST).remove_dots().lowercase(),
        Datasets(TEST).lowercase().remove_dots(),
        Datasets(TEST).remove_dots().lemmatize(),
        Datasets(TEST).lemmatize().remove_dots(),
        Datasets(TEST).remove_stop_words().lowercase(),
        Datasets(TEST).lowercase().remove_stop_words(),
        Datasets(TEST).remove_stop_words().lemmatize(),
        Datasets(TEST).lemmatize().remove_stop_words(),
        Datasets(TEST).remove_dots().remove_stop_words().lowercase(),
        Datasets(TEST).remove_dots().lowercase().remove_stop_words(),
        Datasets(TEST).remove_stop_words().remove_dots().lowercase(),
        Datasets(TEST).remove_dots().remove_stop_words().lemmatize(),
        Datasets(TEST).remove_dots().lemmatize().remove_stop_words(),
        Datasets(TEST).remove_stop_words().remove_dots().lemmatize(),
        Datasets(TEST).remove_stop_words().lemmatize().remove_dots(),
        Datasets(TEST).lemmatize().remove_stop_words().remove_dots(),
        Datasets(TEST).remove_dots().remove_stop_words().lowercase().lemmatize(),
        Datasets(TEST).remove_dots().remove_stop_words().lemmatize().lowercase(),
        Datasets(TEST).remove_dots().lowercase().remove_stop_words().lemmatize(),
        Datasets(TEST).remove_stop_words().remove_dots().lowercase().lemmatize(),
        Datasets(TEST).remove_stop_words().lowercase().remove_dots().lemmatize(),
        Datasets(TEST).lowercase().remove_stop_words().remove_dots().lemmatize(),
        Datasets(TEST).remove_dots().remove_stop_words().lowercase().lemmatize()
    ]



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