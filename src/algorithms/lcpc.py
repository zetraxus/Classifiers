from itertools import combinations

import numpy as np
from numpy import ndarray

from src.algorithm import Algorithm


class LCPC(Algorithm):
    data: ndarray

    def __init__(self):
        super().__init__()
        self.threshold = 0.05
        self.class_distr = dict()
        self.column_info = None
        self.most_popular_class = None
        self.powers = None

    def train(self, ds, column_info):
        self.data = np.array(ds)
        self.column_info = column_info
        self.powers = [2 ** i for i in range(len(self.data[0]))]

        for row in self.data:
            cl = row[-1]
            if cl not in self.class_distr:
                self.class_distr[cl] = 0
            self.class_distr[cl] += 1

        self.most_popular_class = self.__most_popular_class()

    def predict(self, sample):
        filtered_dataset = self.__filter_dataset(sample)
        new_filtered_dataset = self.__new_filter_dataset(sample)

        results = dict()
        new_results = dict()

        for i in range(1, len(sample) + 1):
            pattern_matches = dict()
            new_pattern_matches = dict()

            patterns = list(self.__generate_combinations(len(sample), i))
            new_patterns = list(self.__new_generate_combinations(len(sample), i))

            for p in patterns:
                pattern_matches = self.__analyse_pattern(p, filtered_dataset, pattern_matches)

            for p in new_patterns:
                new_pattern_matches = self.__new_analyse_pattern(p, new_filtered_dataset, new_pattern_matches)

            for k, v in pattern_matches.items():
                if k not in results.keys():
                    results[k] = v

            for k, v in new_pattern_matches.items():
                if k not in new_results.keys():
                    new_results[k] = v

            if len(results) != len(new_results):
                print("oj")

            if len(results) == len(self.class_distr):
                break

        a = self.__best_match_class(results)
        b = self.__best_match_class(new_results)
        print(a == b)

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
                pattern_matches[cl_name] = max(pattern_matches[cl_name], cl_cnt)

        return pattern_matches

    def __new_analyse_pattern(self, pattern, filtered_dataset, pattern_matches):
        options = list()
        for k, v in filtered_dataset.items():
            if self.__new_check_pattern(pattern, k):
                options.append(k)

        if options:
            success, cl_name, cl_cnt = self.__check_options(filtered_dataset, options)
            if success:
                if cl_name not in pattern_matches:
                    pattern_matches[cl_name] = 0
                pattern_matches[cl_name] = max(pattern_matches[cl_name], cl_cnt)

        return pattern_matches

    def __generate_combinations(self, length, count):
        for positions in combinations(range(length), count):
            p = ["0"] * length
            for i in positions:
                p[i] = "1"
            a = 0
            yield ''.join(p)

    def __new_generate_combinations(self, length, count):
        for positions in combinations(range(length), count):
            p = 0
            for i in positions:
                p += self.powers[i]
            yield p

    def __filter_dataset(self, sample):
        filtered_dataset = dict()
        for i in range(self.data.shape[0]):
            row = ""
            for j in range(len(sample)):
                row = self.__check_cell(sample, i, j, row)

            if '1' in row:
                if row not in filtered_dataset:
                    filtered_dataset[row] = dict()
                if self.data[i][-1] not in filtered_dataset[row]:
                    filtered_dataset[row][self.data[i][-1]] = 0
                filtered_dataset[row][self.data[i][-1]] += 1
        return filtered_dataset

    def __new_filter_dataset(self, sample):
        filtered_dataset = dict()
        for i in range(self.data.shape[0]):
            row = 0
            for j in range(len(sample)):
                if self.__new_check_cell(sample, i, j):
                    row += self.powers[j]

            if row > 0:
                if row not in filtered_dataset:
                    filtered_dataset[row] = dict()
                if self.data[i][-1] not in filtered_dataset[row]:
                    filtered_dataset[row][self.data[i][-1]] = 0
                filtered_dataset[row][self.data[i][-1]] += 1
        return filtered_dataset

    def __check_cell(self, sample, i, j, row):
        diff = self.threshold * self.column_info[0][j]['diff'] if self.column_info[1][j] == 'nc' else 0
        if float(self.data[i][j]) - diff <= sample[j] <= float(self.data[i][j]) + diff:
            return f'{row}1'
        else:
            return f'{row}0'

    def __new_check_cell(self, sample, i, j):
        diff = self.threshold * self.column_info[0][j]['diff'] if self.column_info[1][j] == 'nc' else 0
        if float(self.data[i][j]) - diff <= sample[j] <= float(self.data[i][j]) + diff:
            return True
        else:
            return False

    @staticmethod
    def __check_pattern(pattern, sample):
        for i in range(len(pattern)):
            if pattern[i] == '1':
                if sample[i] != '1':
                    return False
        return True

    @staticmethod
    def __new_check_pattern(pattern, sample):
        return pattern & sample == pattern

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
                    class_cnt += cnt
                    class_name = cl
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
            rating.sort(key=lambda tup: tup[1], reverse=True)
            return rating[0][0]
        else:
            return self.most_popular_class

    def __str__(self):
        return self.__class__.__name__
