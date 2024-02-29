from data_parser import get_train_dataset, get_test_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from data import StatsData, TrainData
from analytics.analytics import Analyzer, benchmark_models
from storage_manager import StorageManager

if __name__ == "__main__":
    print(benchmark_models([NaiveBayes(get_train_dataset()), BaselineRandom()]))
