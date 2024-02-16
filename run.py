from data_parser import get_train_dataset, get_test_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from analytics.analyse import Analytics
from data import StatsData, TrainData
from storage_manager import StorageManager

if __name__ == "__main__":
    # Storage manager example
    sm = StorageManager()

    nb = NaiveBayes(get_train_dataset())

    nb.test(get_test_dataset()["text"])
    nb_training_data = sm.load_data(str(nb), "train")

    nb_stats_data = sm.store_data(
        StatsData(str(nb), 0.5, 0, 5, 0.5, 0, 0, 0, 0)),

    print(sm.load_data(str(nb), "stats"))

    # bs_random = BaselineRandom(get_test_dataset())

    # baseResults = bs_random.test()
    # anBS = Analytics(baseResults, get_test_dataset())
    # results = nb.test(get_test_dataset()["text"])

    # print("base f1:", anBS.f1_score())

    # an = Analytics(results, get_test_dataset())

    # print("F1:", an.f1_score())
