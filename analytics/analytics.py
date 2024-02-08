from datasets import Dataset


def f1_score():
    return


def calculate_precision():
    return


def calculate_recall():
    return


def calculate_false_positives(test_result_labels, result_labels):
    false_positives = 0

    for i in range(len(test_result_labels)):
        if test_result_labels[i] == "OFF" and result_labels[i] != "OFF":
            false_positives += 1

    return false_positives


def calculate_false_negatives(test_result_labels, result_labels):
    false_negatives = 0

    for i in range(len(test_result_labels)):
        if test_result_labels[i] == "NOT" and result_labels[i] != "NOT":
            false_negatives += 1

    return false_negatives


def calculate_true_positives(test_result_labels, result_labels):
    true_positives = 0

    for i in range(len(test_result_labels)):
        if test_result_labels[i] == "OFF" and result_labels[i] == "OFF":
            true_positives += 1

    return true_positives


def calculate_true_negatives(test_result_labels, result_labels):
    true_negatives = 0

    for i in range(len(test_result_labels)):
        if test_result_labels[i] == "NOT" and result_labels[i] == "NOT":
            true_negatives += 1

    return true_negatives
