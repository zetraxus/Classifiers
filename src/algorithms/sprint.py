from copy import deepcopy
from enum import Enum

from src.algorithm import Algorithm


class Attribute(Enum):
    NUMERIC = 'nc'
    NUMERIC_DISCRETE = 'nd'
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
        return SPRINT.is_criterion_achieved(value, self.criterion)

    def has_children(self) -> bool:
        return self.left is not None and self.right is not None


class SPRINT(Algorithm):
    def __init__(self, min_elements=None):
        super().__init__()
        self.data_types = None
        self.min_elements = min_elements
        self.tree = None

    def train(self, ds, column_info):
        self.data_types = column_info[1]
        self.tree = self.__partition(ds)

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
        if criterion is None:
            clazz = SPRINT.__get_label_class(grouped_data_set)
            return Node(clazz)

        node_positive = self.__partition(subset_positive)
        node_negative = self.__partition(subset_negative)

        return Node(None, criterion, node_positive, node_negative)

    @staticmethod
    def __get_label_class(grouped_data_set):
        sorted_grouped_data_set = deepcopy(grouped_data_set)
        sorted_grouped_data_set.sort(key=lambda data_set: len(data_set))
        return sorted_grouped_data_set[0][0]

    def __choose_criterion_and_subsets(self, data_set) -> (Criterion, list, list):
        gini_set = []
        for attr_idx, attr_type in enumerate(self.data_types):
            criteria = SPRINT.__get_criteria(data_set, attr_idx, attr_type)
            if len(criteria) == 0:
                continue
            for criterion_value in criteria:
                criterion = Criterion(attr_idx, attr_type, criterion_value)
                gini = self.__calculate_gini_split(data_set, criterion)
                gini_set.append((criterion, gini))
        if len(gini_set) == 0:
            return None, None, None
        min_gini = min(gini_set, key=lambda it: it[1])
        subsets = SPRINT.__split_data_set(data_set, min_gini[0])
        return min_gini[0], subsets[0], subsets[1]

    @staticmethod
    def __get_criteria(data_set, attr_index, attr_type):
        unique_values = list(set(data[attr_index] for data in data_set))
        unique_values.sort()
        if attr_type != Attribute.ENUMERIC:
            return unique_values[:-1]

        return unique_values

    @staticmethod
    def __calculate_gini_split(data_set, criterion):
        left, right = SPRINT.__split_data_set(data_set, criterion)
        left_gini = SPRINT.__calculate_gini_index(left)
        right_gini = SPRINT.__calculate_gini_index(right)

        left_element = len(left) / len(data_set) * left_gini
        right_element = len(left) / len(data_set) * right_gini
        return left_element + right_element

    @staticmethod
    def __calculate_gini_index(data_set):
        grouped_data_set = SPRINT.__get_grouped_data_set(data_set)
        gini = 1
        for (clazz, data) in grouped_data_set:
            p = len(data) / len(data_set)
            gini -= p ** 2

        return gini

    @staticmethod
    def __get_grouped_data_set(data_set):
        values = set(x[-1] for x in data_set if x[-1])
        return [(x, [y for y in data_set if y[-1] == x]) for x in values]

    @staticmethod
    def __split_data_set(data_set, criterion) -> (list, list):
        positive, negative = [], []
        for data in data_set:
            is_achieved = SPRINT.is_criterion_achieved(data, criterion)
            if is_achieved:
                positive.append(data)
            else:
                negative.append(data)
        return positive, negative

    @staticmethod
    def is_criterion_achieved(value, criterion) -> bool:
        if criterion.attribute_type == Attribute.ENUMERIC:
            return value[criterion.attribute_index] in criterion.value

        return value[criterion.attribute_index] <= criterion.value

    def __str__(self):
        return self.__class__.__name__ + " " + str(self.min_elements)
