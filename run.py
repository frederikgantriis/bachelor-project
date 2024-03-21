from data_parser import Datasets
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.baseline_random import BaselineRandom
from benchmarker import Benchmarker
from constants import TRAIN, TEST

if __name__ == "__main__":
    dataset_train = Datasets(TRAIN)
    dataset_train_remove_dots = Datasets(TRAIN).remove_dots()
    dataset_train_remove_stop_words = Datasets(TRAIN).remove_stop_words()
    dataset_train_lowercase = Datasets(TRAIN).lowercase()
    dataset_train_lemmatize = Datasets(TRAIN).lemmatize()
    dataset_train_remove_dots_remove_stop_words = (
        Datasets(TRAIN).remove_dots().remove_stop_words()
    )
    dataset_train_remove_dots_lowercase = Datasets(TRAIN).remove_dots().lowercase()
    dataset_train_remove_dots_lemmatize = Datasets(TRAIN).remove_dots().lemmatize()
    dataset_train_remove_stop_words_lowercase = (
        Datasets(TRAIN).remove_stop_words().lowercase()
    )
    dataset_train_remove_stop_words_lemmatize = (
        Datasets(TRAIN).remove_stop_words().lemmatize()
    )
    dataset_train_lowercase_lemmatize = Datasets(TRAIN).lowercase().lemmatize()
    dataset_train_remove_dots_remove_stop_words_lowercase = (
        Datasets(TRAIN).remove_dots().remove_stop_words().lowercase()
    )
    dataset_train_remove_dots_remove_stop_words_lemmatize = (
        Datasets(TRAIN).remove_dots().remove_stop_words().lemmatize()
    )
    dataset_train_remove_dots_lowercase_lemmatize = (
        Datasets(TRAIN).remove_dots().lowercase().lemmatize()
    )
    dataset_train_remove_stop_words_lowercase_lemmatize = (
        Datasets(TRAIN).remove_stop_words().lowercase().lemmatize()
    )
    dataset_train_remove_dots_remove_stop_words_lowercase_lemmatize = (
        Datasets(TRAIN).remove_dots().remove_stop_words().lowercase().lemmatize()
    )

    dataset_test = Datasets(TEST)
    dataset_test_remove_dots = Datasets(TEST).remove_dots()
    dataset_test_remove_stop_words = Datasets(TEST).remove_stop_words()
    dataset_test_lowercase = Datasets(TEST).lowercase()
    dataset_test_lemmatize = Datasets(TEST).lemmatize()
    dataset_test_remove_dots_remove_stop_words = (
        Datasets(TEST).remove_dots().remove_stop_words()
    )
    dataset_test_remove_dots_lowercase = Datasets(TEST).remove_dots().lowercase()
    dataset_test_remove_dots_lemmatize = Datasets(TEST).remove_dots().lemmatize()
    dataset_test_remove_stop_words_lowercase = (
        Datasets(TEST).remove_stop_words().lowercase()
    )
    dataset_test_remove_stop_words_lemmatize = (
        Datasets(TEST).remove_stop_words().lemmatize()
    )
    dataset_test_lowercase_lemmatize = Datasets(TEST).lowercase().lemmatize()
    dataset_test_remove_dots_remove_stop_words_lowercase = (
        Datasets(TEST).remove_dots().remove_stop_words().lowercase()
    )
    dataset_test_remove_dots_remove_stop_words_lemmatize = (
        Datasets(TEST).remove_dots().remove_stop_words().lemmatize()
    )
    dataset_test_remove_dots_lowercase_lemmatize = (
        Datasets(TEST).remove_dots().lowercase().lemmatize()
    )
    dataset_test_remove_stop_words_lowercase_lemmatize = (
        Datasets(TEST).remove_stop_words().lowercase().lemmatize()
    )
    dataset_test_remove_dots_remove_stop_words_lowercase_lemmatize = (
        Datasets(TEST).remove_dots().remove_stop_words().lowercase().lemmatize()
    )

    naive_bayes = NaiveBayes(dataset_train)
    naive_bayes_remove_dots = NaiveBayes(dataset_train_remove_dots)
    naive_bayes_remove_stop_words = NaiveBayes(dataset_train_remove_stop_words)
    naive_bayes_lowercase = NaiveBayes(dataset_train_lowercase)
    naive_bayes_lemmatize = NaiveBayes(dataset_train_lemmatize)
    naive_bayes_remove_dots_remove_stop_words = NaiveBayes(
        dataset_train_remove_dots_remove_stop_words
    )
    naive_bayes_remove_dots_lowercase = NaiveBayes(dataset_train_remove_dots_lowercase)
    naive_bayes_remove_dots_lemmatize = NaiveBayes(dataset_train_remove_dots_lemmatize)
    naive_bayes_remove_stop_words_lowercase = NaiveBayes(
        dataset_train_remove_stop_words_lowercase
    )
    naive_bayes_remove_stop_words_lemmatize = NaiveBayes(
        dataset_train_remove_stop_words_lemmatize
    )
    naive_bayes_lowercase_lemmatize = NaiveBayes(dataset_train_lowercase_lemmatize)
    naive_bayes_remove_dots_remove_stop_words_lowercase = NaiveBayes(
        dataset_train_remove_dots_remove_stop_words_lowercase
    )
    naive_bayes_remove_dots_remove_stop_words_lemmatize = NaiveBayes(
        dataset_train_remove_dots_remove_stop_words_lemmatize
    )
    naive_bayes_remove_dots_lowercase_lemmatize = NaiveBayes(
        dataset_train_remove_dots_lowercase_lemmatize
    )
    naive_bayes_remove_stop_words_lowercase_lemmatize = NaiveBayes(
        dataset_train_remove_stop_words_lowercase_lemmatize
    )
    naive_bayes_remove_dots_remove_stop_words_lowercase_lemmatize = NaiveBayes(
        dataset_train_remove_dots_remove_stop_words_lowercase_lemmatize
    )

    naive_bayes_remove_dots.set_variation_name("remove_dots")
    naive_bayes_remove_stop_words.set_variation_name("remove_stop_words")
    naive_bayes_lowercase.set_variation_name("lowercase")
    naive_bayes_lemmatize.set_variation_name("lemmatize")
    naive_bayes_remove_dots_remove_stop_words.set_variation_name(
        "remove_dots_remove_stop_words"
    )
    naive_bayes_remove_dots_lowercase.set_variation_name("remove_dots_lowercase")
    naive_bayes_remove_dots_lemmatize.set_variation_name("remove_dots_lemmatize")
    naive_bayes_remove_stop_words_lowercase.set_variation_name(
        "remove_stop_words_lowercase"
    )
    naive_bayes_remove_stop_words_lemmatize.set_variation_name(
        "remove_stop_words_lemmatize"
    )
    naive_bayes_lowercase_lemmatize.set_variation_name("lowercase_lemmatize")
    naive_bayes_remove_dots_remove_stop_words_lowercase.set_variation_name(
        "remove_dots_remove_stop_words_lowercase"
    )
    naive_bayes_remove_dots_remove_stop_words_lemmatize.set_variation_name(
        "remove_dots_remove_stop_words_lemmatize"
    )
    naive_bayes_remove_dots_lowercase_lemmatize.set_variation_name(
        "remove_dots_lowercase_lemmatize"
    )
    naive_bayes_remove_stop_words_lowercase_lemmatize.set_variation_name(
        "remove_stop_words_lowercase_lemmatize"
    )
    naive_bayes_remove_dots_remove_stop_words_lowercase_lemmatize.set_variation_name(
        "remove_dots_remove_stop_words_lowercase_lemmatize"
    )

    benchmarker = Benchmarker(
        [
            (naive_bayes, dataset_test),
            (naive_bayes_remove_dots, dataset_test_remove_dots),
            (naive_bayes_remove_stop_words, dataset_test_remove_stop_words),
            (naive_bayes_lowercase, dataset_test_lowercase),
            (naive_bayes_lemmatize, dataset_test_lemmatize),
            (
                naive_bayes_remove_dots_remove_stop_words,
                dataset_test_remove_dots_remove_stop_words,
            ),
            (naive_bayes_remove_dots_lowercase, dataset_test_remove_dots_lowercase),
            (naive_bayes_remove_dots_lemmatize, dataset_test_remove_dots_lemmatize),
            (
                naive_bayes_remove_stop_words_lowercase,
                dataset_test_remove_stop_words_lowercase,
            ),
            (
                naive_bayes_remove_stop_words_lemmatize,
                dataset_test_remove_stop_words_lemmatize,
            ),
            (naive_bayes_lowercase_lemmatize, dataset_test_lowercase_lemmatize),
            (
                naive_bayes_remove_dots_remove_stop_words_lowercase,
                dataset_test_remove_dots_remove_stop_words_lowercase,
            ),
        ]
    )

    print(benchmarker.benchmark_models(10, None))
