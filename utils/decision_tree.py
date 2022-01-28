"""Дерево решений.

TODO: как получить список с названиями колонок?
solution: list(X.columns)
TODO: как получить множество категорий в колонке?
solution: set(X.tolist())
TODO: как узнать индексы тех, строк, что удовлетворяют условию?
solution:
"""
import math


class DecisionTree:
    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.class_names = set(y.tolist())

        # for feature_name in self.feature_names:
        #     if X[feature_name].dtype != 'float64':
        #         pass

        entropy = 0
        amount = X.shape[0]
        for cls in self.class_names:
            cls_amount = y[cls].shape[0]
            entropy -= cls_amount/amount * math.log10(cls_amount/amount)
        print(entropy)


class Node:
    def __init__(self):
        pass
