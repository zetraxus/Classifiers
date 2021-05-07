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
                    filtered_dataset[row] = dict()
                if self.data[i][-1] not in filtered_dataset[row]:
                    filtered_dataset[row][self.data[i][-1]] = 0
                filtered_dataset[row][self.data[i][-1]] += 1

        min_cp = dict()
        for k, v in filtered_dataset.items():
            if len(v) == 1:
                cl, cnt = list(v.keys())[0], list(v.values())[0]
                if cl not in min_cp:
                    min_cp[cl] = []
                min_cp[cl].append((k, cnt))
        if len(min_cp) > 1:
            a = 0


class SPRINT(Algorithm):
    def __init__(self):
        super().__init__()

    def train(self, ds):
        pass

    def predict(self, sample):
        pass