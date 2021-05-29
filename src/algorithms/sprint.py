from enum import Enum

import numpy as np
from numpy import ndarray

from src.algorithm import Algorithm


class Attribute(Enum):
    NUMERIC = 'n'
    ENUMERIC = 'e'


class Criterion:
    def __init__(self, attribute_index, attribute_type, value):
        self.attribute_index = attribute_index
        self.attribute_type = attribute_type
        self.value = value


class Node:
    def __init__(self, clazz, criterion=None, left=None, right=None):
        self.clazz = clazz
        self.criterion = criterion
        self.left = left
        self.right = right

    def is_criterion_achieved(self, value) -> bool:
        return SPRINT.is_criterion_achieved(value, self.criterion.criterion)

    def has_children(self) -> bool:
        return self.left is not None and self.right is not None


class SPRINT(Algorithm):
    def __init__(self, data_types, min_elements=None):
        super().__init__()
        self.data_types = data_types
        self.min_elements = min_elements
        self.tree = None

    def train(self, ds):
        data_set = np.array(ds)
        self.tree = self.__partition(data_set)

    def predict(self, sample):
        node = self.tree
        while node.has_children():
            if node.is_criterion_achieved(sample):
                node = node.left
            else:
                node = node.right

        return node.clazz

    def __partition(self, data_set) -> Node:
        is_not_enough_items = False if self.min_elements is None else len(data_set) <= self.min_elements
        grouped_data_set = SPRINT.__get_grouped_data_set(data_set)
        if len(grouped_data_set) == 1 or is_not_enough_items:
            clazz = SPRINT.__get_label_class(grouped_data_set)
            return Node(clazz)

        (criterion, subset_positive, subset_negative) = self.__choose_criterion_and_subsets(data_set)
        node_positive = self.__partition(subset_positive)
        node_negative = self.__partition(subset_negative)

        return Node(None, criterion, node_positive, node_negative)

    @staticmethod
    def __get_grouped_data_set(data_set):
        values = set(x[-1] for x in data_set if x[-1])
        return [(x, [y for y in data_set if y[-1] == x]) for x in values]

    @staticmethod
    def __get_label_class(grouped_data_set):
        sorted_grouped_data_set = grouped_data_set.sort(key=lambda data_set: len(data_set))
        return sorted_grouped_data_set[0][0]

    def __choose_criterion_and_subsets(self, data_set) -> (Criterion, ndarray, ndarray):
        gini_set = []
        for attr_idx, attr_type in enumerate(self.data_types):
            criteria = SPRINT.__get_criteria(data_set, attr_idx, attr_type)
            for criterion_value in criteria:
                criterion = Criterion(attr_idx, attr_type, criterion_value)
                gini = self.__calculate_gini_index(data_set, criterion_value)
                gini_set.append((criterion, gini))
        min_gini = min(gini_set, key=lambda it: it[1])
        subsets = SPRINT.__split_data_set(data_set, min_gini[0])
        return min_gini[0], subsets[0], subsets[1]

    @staticmethod
    def __get_criteria(data_set, attr_index, attr_type):
        unique_values = set(data[attr_index] for data in data_set)
        if attr_type == Attribute.NUMERIC:
            return list(unique_values[:-1])

        return list(unique_values)

    def __calculate_gini_index(self, data_set, criterion):
        subsets = SPRINT.__split_data_set(data_set, criterion)
        return -1 # todo

    @staticmethod
    def __split_data_set(data_set, criterion) -> (ndarray, ndarray):
        positive, negative = [], []
        for data in data_set:
            is_achieved = SPRINT.is_criterion_achieved(data[criterion.attribute_index], criterion)
            if is_achieved:
                positive.append(data)
            else:
                negative.append(data)
        return positive, negative

    @staticmethod
    def is_criterion_achieved(value, criterion) -> bool:
        if criterion.attribute_type == Attribute.NUMERIC:
            return value <= criterion.value

        return value in criterion.value

    def __str__(self):
        return self.__class__.__name__ + " " + str(self.min_elements)
