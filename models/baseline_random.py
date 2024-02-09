import random
from analytics.constants import NOT, OFF


def test(comments):
    labels = [NOT, OFF]

    answer = []

    for _ in range(len(comments)):
        answer.append(random.choice(labels))

    return answer
