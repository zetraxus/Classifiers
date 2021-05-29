import random
from math import floor


class Dataset:
    def __init__(self, input, split_ratio, column_types, buckets):
        self.data, self.data_in_buckets, self.info = [], [], []
        self.column_types = column_types

        self.__prepare_data(input, column_types)
        random.shuffle(self.data)

        self.__split_data_into_buckets(buckets)
        self.dataset_size = len(self.data)
        self.split_ratio = split_ratio

    def __prepare_data(self, input, column_types):
        for i in range(len(column_types)):
            self.info.append(dict())

        for row in input:
            for i in range(len(row) - 1):
                if column_types[i] == 'e':
                    if row[i] not in self.info[i]:
                        self.info[i][row[i]] = len(self.info[i])
                    row[i] = self.info[i][row[i]]
                else:
                    elem = float(row[i])
                    if 'min' not in self.info[i] or elem < self.info[i]['min']:
                        self.info[i]['min'] = elem
                    if 'max' not in self.info[i] or elem > self.info[i]['max']:
                        self.info[i]['max'] = elem
                    row[i] = elem

            self.data.append(row)

        for i in range(len(self.info)):
            if self.column_types[i] == 'nc':
                self.info[i]['diff'] = self.info[i]['max'] - self.info[i]['min']

    def __split_data_into_buckets(self, buckets):
        for row in self.data:
            next_row = []
            for i in range(len(row) - 1):
                if self.column_types[i] != 'e':
                    _min = self.info[i]['min']
                    _max = self.info[i]['max']
                    bucket = floor(buckets * (row[i] - _min) / (_max - _min))
                    if bucket >= buckets:  # occurs only when row[i] == max column_value
                        bucket = buckets - 1
                    next_row.append(bucket)
                else:
                    next_row.append(row[i])
            next_row.append(row[len(row) - 1])
            self.data_in_buckets.append(next_row)

    def get_train_set(self, buckets=False):
        split_position = floor(self.dataset_size * self.split_ratio)
        if buckets:
            return self.data_in_buckets[:split_position]
        else:
            return self.data[:split_position]

    def get_test_set(self, buckets=False):
        split_position = floor(self.dataset_size * self.split_ratio)
        if buckets:
            return self.data_in_buckets[split_position:]
        else:
            return self.data[split_position:]

    def get_info(self):
        return self.info, self.column_types
