from enum import Enum

import numpy as np

from src.algorithm import Algorithm


class Attribute(Enum):
    NUMERIC = 1
    ENUMERIC = 2


class Node:
    def __init__(self, clazz, criterion_type=None, attribute_index=None, criterion=None, left=None, right=None):
        self.clazz = clazz
        self.attribute_type = criterion_type
        self.attribute_index = attribute_index
        self.criterion = criterion
        self.left = left
        self.right = right

    def is_criterion_achieved(self, value) -> bool:
        if self.attribute_type == Attribute.NUMERIC:
            return value <= self.criterion

        return value in self.criterion

    def has_children(self) -> bool:
        return self.left is not None and self.right is not None


class SPRINT(Algorithm):
    def __init__(self, min_elements=None):
        super().__init__()
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
        grouped_data_set = self.__get_grouped_data_set(data_set)
        if len(grouped_data_set) == 1 or is_not_enough_items:
            clazz = self.__get_label_class(grouped_data_set)
            return Node(clazz)

        (criterion, subset_positive, subset_negative) = self.__choose_criterion_and_subsets(data_set)
        node_positive = self.__partition(subset_positive)
        node_negative = self.__partition(subset_negative)

        return Node(None, attribute_type, attribute_index, criterion, node_positive, node_negative)

    def __get_grouped_data_set(self, data_set):
        values = set(x[-1] for x in data_set if x[-1])
        return [(x, [y for y in data_set if y[-1] == x]) for x in values]

    def __get_label_class(self, grouped_data_set):
        sorted_grouped_data_set = grouped_data_set.sort(key=lambda data_set: len(data_set))
        return sorted_grouped_data_set[0][0]

    def __choose_criterion_and_subsets(self, data_set):
        attributes = list(range(len(data_set[0]) - 1))
        pass
