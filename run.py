from data_parser import get_train_dataset, get_test_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes
from analytics.analyse import Analytics

if __name__ == "__main__":
    nb = NaiveBayes(get_train_dataset())
    bs_random = BaselineRandom(get_test_dataset())

    baseResults = bs_random.test(get_test_dataset()["text"])
    anBS = Analytics(baseResults, get_test_dataset())
    results = nb.test(get_test_dataset()["text"])

    print("base f1:", anBS.f1_score())

    an = Analytics(results, get_test_dataset())

    print("F1:", an.f1_score())
