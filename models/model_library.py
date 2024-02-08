import re

labels = ["OFF", "NOT"]


def get_labels():
    return labels

def sanitize(line):
    return re.findall(r'[a-øA-ø0-9-]+|[^a-zæøåA-ZÆØÅ0-9\s]+', line)