from data_parser import get_train_dataset, get_test_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from data import StatsData, TrainData
from analytics.analytics import Analyzer, benchmark_models
from storage_manager import StorageManager

from models.logistic_regression import LogisticRegression

if __name__ == "__main__":    
    print(benchmark_models([NaiveBayes(get_train_dataset()), BaselineRandom()]))

    lr = LogisticRegression(get_train_dataset())
    print(lr.gradident_descent([3, 2], [0,0,0], lr.sigmoid(0) - 1, 0.1))
    