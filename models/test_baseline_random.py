from baseline_random import BaselineRandom
from datasets import DatasetDict, Dataset
from data_parser import get_test_dataset, get_train_dataset

def test_make_new_baseline_random_object():
    data = BaselineRandom(get_train_dataset())
    assert data.dataset == get_train_dataset()

def test_baseline_random_test():
    data = BaselineRandom(get_train_dataset())
    isinstance(data.test(get_test_dataset()), list)

def test_baseline_random_str():
    data = BaselineRandom(get_train_dataset())
    assert str(data) == "Baseline Random"