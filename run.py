from models.baseline_random import BaselineRandom
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from data_parser import get_train_dataset
from analytics.benchmarker import Benchmarker

if __name__ == "__main__":
    print(
        Benchmarker(
            [
                NaiveBayes(get_train_dataset()),
                BaselineRandom(),
                LogisticRegression(get_train_dataset()),
            ]
        ).benchmark_models(40)
    )
