from data_parser import get_train_dataset
from models.baseline_random import BaselineRandom
from models.naive_bayes import NaiveBayes

if __name__ == "__main__":
    nb = NaiveBayes(get_train_dataset())
    bs_random = BaselineRandom(get_train_dataset())

    result = nb.test("hej med dig")
    # result = bs_random.test()

    print(result)
