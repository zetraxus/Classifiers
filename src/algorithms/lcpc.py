from itertools import combinations

import numpy as np
from numpy import ndarray

from src.algorithm import Algorithm


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
        filtered_dataset = self.__filter_dataset(sample)
        pattern_matches = list()
        for i in range(1, len(sample) + 1):
            print(i)
            patterns = list(self.__generate_combinations(len(sample), i))
            for p in patterns:
                options = []
                for k, v in filtered_dataset.items():
                    if self.__check_pattern(p, k):
                        options.append(k)

                if options:
                    succ, cl_name, cl_cnt = self.__check_options(filtered_dataset, options)
                    if succ:
                        pattern_matches.append((cl_name, cl_cnt))
        print('# ', len(pattern_matches))

    @staticmethod
    def __generate_combinations(length, count):
        for positions in combinations(range(length), count):
            p = ["0"] * length
            for i in positions:
                p[i] = "1"
            yield ''.join(p)

    def __filter_dataset(self, sample):
        filtered_dataset = dict()
        for i in range(self.data.shape[0]):
            useful_row, row = False, ""
            for j in range(len(sample)):
                if sample[j] == self.data[i][j]:
                    row, useful_row = f'{row}1', True
                else:
                    row = f'{row}0'

            if useful_row:
                if row not in filtered_dataset:
                    filtered_dataset[row] = dict()
                if self.data[i][-1] not in filtered_dataset[row]:
                    filtered_dataset[row][self.data[i][-1]] = 0
                filtered_dataset[row][self.data[i][-1]] += 1
        return filtered_dataset

    @staticmethod
    def __check_pattern(pattern, sample):
        match = True
        for i in range(len(pattern)):
            if pattern[i] == '1':
                if sample[i] != '1':
                    match = False
                    break
        return match

    @staticmethod
    def __check_options(filtered_dataset, options):
        class_name, class_cnt, success = "", 0, True
        for option in options:
            if len(filtered_dataset[option]) > 1:
                success = False
                break
            else:
                option_dict = filtered_dataset[option]
                cl = list(option_dict.keys())[0]
                cnt = option_dict[cl]
                if class_name or class_name == cl:
                    class_name = cl
                    class_cnt += cnt
                else:
                    success = False
                    break
        return success, class_name, class_cnt
