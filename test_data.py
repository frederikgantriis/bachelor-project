from data import Data, StatsData, TrainData
from pandas import DataFrame


def test_make_new_data_object():
    data = Data("model_name")
    assert data.model_name == "model_name"


def test_make_new_stats_data_object():
    stats_data = StatsData("model_name", 0.9, 90, 0.9, 0.9, 90, 10, 10, 90)
    assert stats_data.model_name == "model_name"
    assert stats_data.f1 == 0.9
    assert stats_data.accuracy == 90
    assert stats_data.precision == 0.9
    assert stats_data.recall == 0.9
    assert stats_data.true_positives == 90
    assert stats_data.false_positives == 10
    assert stats_data.false_negatives == 10
    assert stats_data.true_negatives == 90


def test_make_new_train_data_object():
    train_data = TrainData("model_name", (0, 1))
    assert train_data.model_name == "model_name"
    assert train_data.data == (0, 1)


def test_stats_data_as_data_frame():
    stats_data = StatsData("model_name", 0.9, 90, 0.9, 0.9, 90, 10, 10, 90)
    df = stats_data.as_data_frame()
    assert isinstance(df, DataFrame)
    assert df["model_name"].values[0] == "model_name"
    assert df["f1"].values[0] == 0.9
    assert df["accuracy"].values[0] == 90
    assert df["precision"].values[0] == 0.9
    assert df["recall"].values[0] == 0.9
    assert df["true_positives"].values[0] == 90
    assert df["false_positives"].values[0] == 10
    assert df["false_negatives"].values[0] == 10
    assert df["true_negatives"].values[0] == 90
