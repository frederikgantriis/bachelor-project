from datasets import Dataset


def f1_score():
    return


def calculate_precision(test_result_labels, result_labels):
    true_positives = calculate_true_positives(
        test_result_labels, result_labels)

    return true_positives / (true_positives + calculate_true_negatives(test_result_labels, result_labels))


def calculate_recall(test_result_labels, result_labels):
    true_positives = calculate_true_positives(
        test_result_labels, result_labels)

    return true_positives / (true_positives + calculate_false_negatives(test_result_labels, result_labels))


def calculate_false_positives(test_result_labels, result_labels):
    return count_true_labels(test_result_labels, result_labels, "OFF", "NOT")


def calculate_false_negatives(test_result_labels, result_labels):
    return count_true_labels(test_result_labels, result_labels, "NOT", "OFF")


def calculate_true_positives(test_result_labels, result_labels):
    return count_true_labels(test_result_labels, result_labels, "OFF", "OFF")


def calculate_true_negatives(test_result_labels, result_labels):
    return count_true_labels(test_result_labels, result_labels,"NOT", "NOT")


def count_true_labels(test_result_labels, result_labels, compare_test_label, compare_result_label):
    counter = 0

    for i in range(len(test_result_labels)):
        if test_result_labels[i] == compare_test_label and result_labels[i] == compare_result_label:
            counter += 1

    return counter
