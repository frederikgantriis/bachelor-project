from data_parser import Datasets
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes

if __name__ == "__main__":
    dataset_train = Datasets("train")
    dataset_train.remove_dots()
    
    dataset_test = Datasets("test")
    dataset_test.remove_dots()
    nb1 = NaiveBayes(dataset_train)
    nb1.test(dataset_test.to_list())
    lr = LogisticRegression(dataset_train)
    lr.test(dataset_test.to_list())