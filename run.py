from data_parser import Datasets

if __name__ == "__main__":
    dataset_train = Datasets("train")
    dataset_train.remove_dots()
    print(dataset_train.to_dict()["OFF"][0])
    dataset_train.remove_stop_words()
    print(dataset_train.to_dict()["OFF"][0])
    dataset_train.remove_stop_words().remove_dots()
    print(dataset_train.to_dict()["OFF"][0])
