from analytics.analyse import Analytics
from constants import NOT, OFF


"""
This test_case should generate:
True Positive: 2
True Negative: 1
False Positive: 2
False Negative: 1
"""
test_result = [NOT, OFF, NOT, OFF, OFF, OFF]

test_dataset = {}

test_dataset["labels"] = [NOT, OFF, OFF, NOT, OFF, NOT]

test_analytics = Analytics(test_result, test_dataset)


def test_calculate_true_positives():
    assert test_analytics.calculate_true_positives() == 2


def test_calculate_true_negative():
    assert test_analytics.calculate_true_negatives() == 1


def test_calculate_false_positives():
    assert test_analytics.calculate_false_positives() == 2


def test_calculate_false_negatives():
    assert test_analytics.calculate_false_negatives() == 1


def test_calculate_precision():
    assert test_analytics.calculate_precision() == 2 / 3


def test_calculate_recall():
    assert test_analytics.calculate_recall() == 2 / 3


def test_f1_score():
    assert test_analytics.f1_score() == 4 / 7
