from data_parser import get_test_comments, get_test_labels, get_train
from models import baseline_random
from models.naive_bayes import NaiveBayes


def test_random():
    test_comments = get_test_comments()

    result = baseline_random.test(test_comments)

    return compare_with_test_data(result)


def compare_with_test_data(results):
    test_labels = get_test_labels()

    correct_results = 0

    for i in range(len(results)):
        if test_labels[i] == results[i]:
            correct_results += 1

    return 100 * (correct_results/len(test_labels))


if __name__ == "__main__":
    nb = NaiveBayes(get_train())

    result = nb.test("hej med dig")

    print(result)
