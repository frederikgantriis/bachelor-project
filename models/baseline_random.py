import random
import utils


def test(comments):
    labels = utils.get_labels()

    answer = []

    for _ in range(len(comments)):
        answer.append(random.choice(labels))

    return answer
