import numpy as np
from numpy import ndarray


class Algorithm:
    def __init__(self):
        pass

    def train(self, ds):
        pass

    def predict(self, sample):
        pass


class NaiveBayes(Algorithm):
    def __init__(self):
        super().__init__()

    def train(self, ds):
        pass

    def predict(self, sample):
        pass


class LCPC(Algorithm):
    data: ndarray

    def __init__(self):
        super().__init__()
        self.class_distr = dict()

    def train(self, ds):
        self.data = np.array(ds)
        for row in self.data:
            cl = row[-1]
            if cl not in self.class_distr:
                self.class_distr[cl] = 1
            else:
                self.class_distr[cl] += 1

    def predict(self, sample):
        filtered_dataset = dict()

        for i in range(self.data.shape[0]):
            useful_row, row = False, ""
            for j in range(len(sample)):
                if sample[j] == self.data[i][j]:
                    row = f'{row}1'
                    useful_row = True
                else:
                    row = f'{row}0'
            if useful_row:
                if row not in filtered_dataset:
                    filtered_dataset[row] = set()

                filtered_dataset[row].add(self.data[i][-1])


class SPRINT(Algorithm):
    def __init__(self):
        super().__init__()

    def train(self, ds):
        pass

    def predict(self, sample):
        pass
