from data_parser import Datasets
from models.naive_bayes import NaiveBayes

if __name__ == "__main__":
    dataset_train = Datasets("train")
    dataset_train.remove_dots()
    print(dataset_train.to_dict()["OFF"][0])
    dataset_train.remove_stop_words()
    print(dataset_train.to_dict()["OFF"][0])

    dataset_train.remove_stop_words().remove_dots()
    print(dataset_train.to_dict()["OFF"][0])

    nb1 = NaiveBayes(dataset_train)
    dataset_test = Datasets("test")
    dataset_test.remove_dots()
    nb1.test(dataset_test.to_list())
    