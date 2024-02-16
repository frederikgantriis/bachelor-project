from data_parser import get_train_dataset, get_test_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from analytics.analytics import Analyzer
from data import TestData, TrainData
from storage_manager import StorageManager

if __name__ == "__main__":
    # nb = NaiveBayes(get_train_dataset())
    # nb.train()
    # print(nb.test())
    test = TestData("test", 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0)

    sm = StorageManager(test)
    sm.store_data()
    print(sm.load_data())

    # bs_random = BaselineRandom(get_test_dataset())

    # baseResults = bs_random.test()
    # anBS = Analytics(baseResults, get_test_dataset())
    # results = nb.test(get_test_dataset()["text"])

    # print("base f1:", anBS.f1_score())

    # an = Analytics(results, get_test_dataset())

    # print("F1:", an.f1_score())
