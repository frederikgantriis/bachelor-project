
from benchmarker import Benchmarker
from data_parser import Datasets
from models.svm import SVM


if __name__ == "__main__":
    # # remove dots lowercase remove stop words lemmatize
    # train_data2 = Datasets("train").remove_dots(
    # ).lowercase().remove_stop_words().lemmatize()

    # test_data2 = Datasets("test").remove_dots(
    # ).lowercase().remove_stop_words().lemmatize()

    train_data = Datasets("train").remove_dots(
    ).lemmatize().remove_stop_words()

    test_data = Datasets("test").remove_dots(
    ).lemmatize().remove_stop_words()

    svm = SVM(train_data)
    svm.train()

    bm = Benchmarker([(svm, test_data)], 10)

    print(bm.benchmark_models())
