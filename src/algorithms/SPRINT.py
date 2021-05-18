import numpy as np
from numpy import ndarray

from Interface import Algorithm


class Node:
    def __init__(self, clazz, criterion, left, right):
        self.clazz = clazz
        self.criterion = criterion
        self.left = left
        self.right = right


class SPRINT(Algorithm):
    def __init__(self, min_elements=None):
        super().__init__()
        self.min_elements = min_elements
        self.tree = None

    def train(self, ds):
        data_set = np.array(ds)
        self.tree = self.partition(data_set)

    def predict(self, sample):
        pass

    def partition(self, data_set) -> Node:
        if self.__check_classes_homogeneity(data_set) or len(data_set) <= self.min_elements:
            clazz = data_set[0][-1]  # todo: assign proper class when min_elements condition was achieved
            return Node(clazz, None, None, None)

        (criterion, subset_positive, subset_negative) = self.__choose_criterion_and_subsets(data_set)
        node_positive = self.partition(subset_positive)
        node_negative = self.partition(subset_negative)

        return Node(None, criterion, node_positive, node_negative)

    def __check_classes_homogeneity(self, data_set) -> bool:
        first_element_class = data_set[0][-1]
        for row in data_set:
            if row[-1] != first_element_class:
                return False

        return True

    def __choose_criterion_and_subsets(self, data_set) -> (str, ndarray, ndarray):
        pass
