import random
from math import floor


class Dataset:
    def __init__(self, input, split_ratio, info, buckets):
        self.data = []

        possibilities = []
        for i in range(len(info)):
            possibilities.append(dict())

        for row in input:
            row_without_strings = row
            for i in range(len(row) - 1):
                if info[i] == 'e':
                    if row[i] not in possibilities[i]:
                        possibilities[i][row[i]] = len(possibilities[i])
                    row_without_strings[i] = possibilities[i][row[i]]
                else:
                    if 'min' not in possibilities[i] or row[i] < possibilities[i]['min']:
                        possibilities[i]['min'] = row[i]
                    if 'max' not in possibilities[i] or row[i] > possibilities[i]['max']:
                        possibilities[i]['max'] = row[i]

            self.data.append(row_without_strings)

        self.info = info
        self.dataset_size = len(self.data)
        self.split_ratio = split_ratio
        random.shuffle(self.data)

    def getTrainSet(self):
        return self.data[:floor(self.dataset_size * self.split_ratio)]

    def getTestSet(self):
        return self.data[floor(self.dataset_size * self.split_ratio):]
