from data_parser import Datasets
from models.naive_bayes import NaiveBayes


if __name__ == "__main__":
    dataset_train = Datasets("train")
    print(dataset_train.unsanitized().to_dict()["OFF"][0])
    print(dataset_train.unsanitized().without_stop_words().to_dict()["OFF"][0])
    print(dataset_train.unsanitized().to_dict()["OFF"][0])

    # nb_type_2 = NaiveBayes(datasets.without_dots("train"))

    # # print(datasets.standard("test"))

    # # datasets: dict[list[str]] = Datasets()

    # # nb_1 = NaiveBayesVariant1(datasets.without_dots)
    # # nb_2 = NaiveBayes(datasets.all_lower)

    # nb_benchmarker = Benchmarker([nb_1, nb_2])
    # nb_benchmarker.draw_graph()
    # nb_benchmarker.draw_cool_graph()
