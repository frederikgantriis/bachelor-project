import random
from models.model_library import get_labels


def test(comments):
    labels = get_labels()

    answer = []

    for _ in range(len(comments)):
        answer.append(random.choice(labels))

    return answer
