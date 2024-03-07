from models.baseline_random import BaselineRandom
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from data_parser import get_train_dataset
from analytics.analytics import benchmark_models

if __name__ == "__main__":    
    print(benchmark_models([NaiveBayes(get_train_dataset()), BaselineRandom(), LogisticRegression(get_train_dataset())]))    