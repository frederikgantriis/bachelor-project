
from benchmarker import Benchmarker
from data_parser import Datasets
from models.svm import SVM


if __name__ == "__main__":
    svm = SVM(Datasets("train"))
    svm.train()
    result = svm.test(Datasets("test").to_list())

    bm = Benchmarker([(svm, Datasets("test"))], 10)

    print(bm.benchmark_models())
