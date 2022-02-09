"""Кастомная реализация дерева решений, которая может работать с категориальными и численными
признаками.
"""
import math

import numpy as np
from graphviz import Digraph  # TODO визуализация дерева графом


class DecisionTree:
    """Дерево решений."""
    def __init__(self, *, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.01):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, Y, categorical_feature_names, numerical_feature_names, special_cases):
        """Обучает дерево решений.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            categorical_feature_names: список категориальных признаков.
            numerical_feature_names: список численных признаков.
            special_cases: ... TODO построить графы зависимости вопросов.
        """
        self.feature_names = list(X.columns)
        self.class_names = set(Y.tolist())
        self.categorical_feature_names = categorical_feature_names
        self.numerical_feature_names = numerical_feature_names
        self.graph = Digraph(comment="моё дерево решений")

        self.tree = self.generate_node(X, Y, self.feature_names)

        return self.tree

    def generate_node(self, X, Y, available_feature_names):
        """Рекурсивная функция создания узлов дерева.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            available_feature_names: список доступных признаков для разбиения входного множества.
        Returns:
            node: узел дерева.
        """
        available_feature_names = available_feature_names.copy()

        best_gain = self.min_impurity_decrease
        best_feature = None
        for feature_name in available_feature_names:
            gain = self._information_gain(X, Y, feature_name)
            if best_gain is None:
                best_gain = gain
                best_feature = feature_name
            elif isinstance(best_gain, float) and isinstance(gain, float):
                if best_gain < gain:
                    best_gain = gain
                    best_feature = feature_name
            elif isinstance(best_gain, tuple) and isinstance(gain, float):
                if best_gain[0] < gain:
                    best_gain = gain
                    best_feature = feature_name
            elif isinstance(best_gain, float) and isinstance(gain, tuple):
                if best_gain < gain[0]:
                    best_gain = gain
                    best_feature = feature_name
            elif isinstance(best_gain, tuple) and isinstance(gain, tuple):
                if best_gain[0] < gain[0]:
                    best_gain = gain
                    best_feature = feature_name

        samples = X.shape[1]
        distribution = dict()
        label = None
        max_samples_per_class = -1
        for class_name in self.class_names:
            samples_per_class = (Y == class_name).sum()
            distribution[class_name] = samples_per_class
            if max_samples_per_class < samples_per_class:
                max_samples_per_class = samples_per_class
                label = class_name

        # TODO:
        # проверка на available_feature_names
        # удаление категориальных признаков
        # и добавление тех, которые не могли быть

        children = []
        if X.shape[0] >= self.min_samples_split:
            xs, ys = self.split(X, Y, best_feature, best_gain[1])

            for x, y in zip(xs, ys):
                children.append(self.generate_node(x, y, available_feature_names))

        node = Node(best_feature, best_gain, samples, distribution, label, children)

        return node

    def split(self, X, Y, feature_name, threshold):
        """Расщепляет множество согласно признаку.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_name: признак, по которому будет происходить расщепление.
            threshold: порог разбиения для численного признака.
        Returns:
            xs: список с подмножествами точек данных.
            ys: список с подмножествами соответствующих меток.
        """
        xs = []
        ys = []
        if feature_name in self.categorical_feature_names:
            for feature_value in set(X[feature_name].tolist()):
                xs.append(X[X[feature_name] == feature_value])
                ys.append(Y[X[feature_name] == feature_value])
        elif feature_name in self.numerical_feature_names:
            x_less = X[X[feature_name] < threshold]
            x_more = X[X[feature_name] >= threshold]
            xs = [x_less, x_more]
            y_less = Y[X[feature_name] < threshold]
            y_more = Y[X[feature_name] >= threshold]
            ys = [y_less, y_more]

        return xs, ys

    def _information_gain(self, X, Y, feature_name):
        """Считает прирост информации для разделения по признаку.

        Формула в LaTeX:
        Gain(A, Q) = H(A, S) -\sum\limits^q_{i=1} \frac{|A_i|}{|A|} H(A_i, S),
        где
        A - множество точек данных;
        Q - метка данных;
        q - множество значений метки, т.е. классы;
        S - признак;
        A_i - множество элементов A, на которых Q имеет значение i.
        """
        # информационный прирост для категориального признака
        if feature_name in self.categorical_feature_names:
            a = self._categorical_feature_entropy(X, feature_name)

            b = 0
            for class_name in self.class_names:
                A_i = (Y == class_name).sum()
                A = X.shape[0]
                x = X[Y == class_name]
                b += (A_i/A) * self._categorical_feature_entropy(x, feature_name)

            gain = a - b

            return gain

        # информационный прирост для численного признака
        elif feature_name in self.numerical_feature_names:

            if X[feature_name].max() is np.NaN:
                print(f'Для признака {feature_name} получилось пустое множество')
                print('---')
                print(X[feature_name])

            # лучший информационный прирост и порог
            best_gain = None
            best_threshold = None
            for threshold in range(int(X[feature_name].max())):
                a = self._numerical_feature_entropy(X, feature_name, threshold)

                b = 0
                for class_name in self.class_names:
                    A_i = (Y == class_name).sum()
                    A = X.shape[0]
                    x = X[Y == class_name]
                    b += (A_i/A) * self._numerical_feature_entropy(x, feature_name, threshold)

                gain = a - b

                if best_gain is None or best_gain < gain:
                    best_gain = gain
                    best_threshold = threshold

            return best_gain, best_threshold

    @staticmethod
    def _categorical_feature_entropy(X, categorical_feature_name):
        """Считает энтропию для категориального признака.

        Формула в LaTeX:
        H(A, S) = -\sum\limits^s_{i = 1} \frac{m_i}{n} \log_2 \frac{m_i}{n},
        где
        A - множество точек данных;
        n - количество точек данных в A;
        S - признак;
        s - множество значений признака S;
        m_i - количество точек данных, имеющих значение s признака S;
        """
        n = X.shape[0]  # количество точек данных в обучающем наборе

        entropy = 0
        # перебор по значениям признака
        for feature_value in set(X[categorical_feature_name].tolist()):
            m_i = (X[categorical_feature_name] == feature_value).sum()
            if m_i != 0:
                entropy -= (m_i/n) * math.log2(m_i/n)

        return entropy

    @staticmethod
    def _numerical_feature_entropy(X, numerical_feature_name, threshold):
        """Считает энтропию для численного признака."""
        n = X.shape[0]  # количество точек данных в обучающем наборе

        less = (X[numerical_feature_name] < threshold).sum()
        more = (X[numerical_feature_name] >= threshold).sum()

        if less == 0 and more == 0:
            entropy = 0
        elif less == 0:
            entropy = -(more/n) * math.log2(more/n)
        elif more == 0:
            entropy = -(less/n) * math.log2(less/n)
        else:
            entropy = -(less/n) * math.log2(less/n) - (more/n) * math.log2(more/n)

        return entropy


class Node:
    """Узел дерева решений."""
    def __init__(self, feature, gain, samples, distribution, label, children):
        self.split = feature
        self.entropy = gain
        self.samples = samples
        self.distribution = distribution
        self.label = label

        self.children = children
