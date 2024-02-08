from analytics import calculate_false_negatives, calculate_false_positives, calculate_true_negatives, calculate_true_positives


"""
This test_case should generate:
True Positive: 2
True Negative: 1
False Positive: 2
False Negative: 1
"""
test_result = ["NOT", "OFF", "NOT", "OFF", "OFF", "OFF"]
result = ["NOT", "OFF", "OFF", "NOT", "NOT", "OFF"]


def test_calculate_true_positives():
    assert calculate_true_positives(test_result, result) == 2


def test_calculate_true_negative():
    assert calculate_true_negatives(test_result, result) == 1


def test_calculate_false_positives():
    assert calculate_false_positives(test_result, result) == 2


def test_calculate_false_negatives():
    assert calculate_false_negatives(test_result, result) == 1
