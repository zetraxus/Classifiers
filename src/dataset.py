import random
from math import floor


class Dataset:
    def __init__(self, input, split_ratio):
        self.data = []
        for row in input:
            self.data.append(row)

        self.dataset_size = len(self.data)
        self.split_ratio = split_ratio
        random.shuffle(self.data)

    def getTrainSet(self):
        # return "first split_ratio * dataset size" elements
        return self.data[:floor(self.dataset_size * self.split_ratio)]

    def getTestSet(self):
        # return last "split_ratio * dataset size" elements
        return self.data[floor(self.dataset_size * self.split_ratio):]
