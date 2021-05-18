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
        results = dict()

        for i in range(1, len(sample) + 1):
            pattern_matches = dict()
            patterns = list(self.__generate_combinations(len(sample), i))
            for p in patterns:
                pattern_matches = self.__analyse_pattern(p, filtered_dataset, pattern_matches)

            for k, v in pattern_matches.items():
                if k not in results.keys():
                    results[k] = v

            if len(results) == len(self.class_distr):
                break

        return self.__best_match_class(results)

    def __analyse_pattern(self, pattern, filtered_dataset, pattern_matches):
        options = list()
        for k, v in filtered_dataset.items():
            if self.__check_pattern(pattern, k):
                options.append(k)

        if options:
            success, cl_name, cl_cnt = self.__check_options(filtered_dataset, options)
            if success:
                if cl_name not in pattern_matches:
                    pattern_matches[cl_name] = 0
                pattern_matches[cl_name] += cl_cnt

        return pattern_matches

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
        for i in range(len(pattern)):
            if pattern[i] == '1':
                if sample[i] != '1':
                    return False
        return True

    @staticmethod
    def __check_options(filtered_dataset, options):
        class_name, class_cnt = "", 0
        for option in options:
            if len(filtered_dataset[option]) > 1:
                return False, None, None
            else:
                option_dict = filtered_dataset[option]
                cl = list(option_dict.keys())[0]
                cnt = option_dict[cl]
                if class_name == "" or class_name == cl:
                    class_name = cl
                    class_cnt += cnt
                else:
                    return False, None, None
        return True, class_name, class_cnt

    def __most_popular_class(self):
        cl, cnt = None, -1
        for k, v in self.class_distr.items():
            if v > cnt:
                cl, cnt = k, v
        return cl

    def __best_match_class(self, results):
        rating = list()
        if len(results):
            for k, v in results.items():
                rating.append((k, v / self.class_distr[k]))
            rating.sort(key=lambda tup: tup[1])
            return rating[0][0]
        else:
            return self.__most_popular_class()
