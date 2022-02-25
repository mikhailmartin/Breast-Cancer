"""Кастомная реализация дерева решений, которая может работать с категориальными и численными
признаками.
"""
import functools
import math

from graphviz import Digraph


def counter(function):
    """Декоратор-счётчик."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return function(*args, **kwargs)
    wrapper.count = 0
    return wrapper


class DecisionTree:
    """Дерево решений."""
    def __init__(
            self,
            *,
            criterion='entropy',
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.05
    ):
        if criterion != 'entropy':
            print('Я пока не умею работать с таким критерием. Посчитаю через энтропию.')
        self.__criterion = 'entropy'
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_impurity_decrease = min_impurity_decrease

        self.__feature_names = None
        self.__class_names = None
        self.__categorical_feature_names = None
        self.__numerical_feature_names = None
        self.__tree = None
        self.__graph = None
        self.__feature_importances_ = None
        self.__total_samples = None

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def class_names(self):
        return self.__class_names

    @property
    def categorical_feature_names(self):
        return self.__categorical_feature_names

    @property
    def numerical_feature_names(self):
        return self.__numerical_feature_names

    @property
    def tree(self):
        return self.__tree

    @property
    def graph(self):
        return self.__graph

    @property
    def feature_importances_(self):
        total = 0
        for importance in self.__feature_importances_.values():
            total += importance
        for feature in self.__feature_importances_.keys():
            self.__feature_importances_[feature] /= total

        return self.__feature_importances_

    def fit(self, X, Y, categorical_feature_names, numerical_feature_names, *, special_cases=None):
        """Обучает дерево решений.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            categorical_feature_names: список категориальных признаков.
            numerical_feature_names: список численных признаков.
            special_cases: словарь {признак, который должен быть первым: признак или список
              признаков, которые могут быть после}.
        """
        self.__feature_names = list(X.columns)
        self.__class_names = sorted(list(set(Y.tolist())))
        self.__categorical_feature_names = categorical_feature_names
        self.__numerical_feature_names = numerical_feature_names

        self.__total_samples = X.shape[0]
        self.__feature_importances_ = dict.fromkeys(self.__feature_names, 0)

        available_feature_names = self.feature_names.copy()
        # удаляем те признаки, которые пока не могут рассматриваться
        if special_cases:
            for value in special_cases.values():
                if isinstance(value, str):
                    available_feature_names.remove(value)
                elif isinstance(value, list):
                    for feature_name in value:
                        available_feature_names.remove(feature_name)

        self.__tree = self.__generate_node(X, Y, None, available_feature_names, special_cases)

    @counter
    def __generate_node(self, X, Y, feature_value, available_feature_names, special_cases=None):
        """Рекурсивная функция создания узлов дерева.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_value: значение признака, по которому этот узел был сформирован.
            available_feature_names: список доступных признаков для разбиения входного множества.
            special_cases: словарь {признак, который должен быть первым: признак, который может быть
              после}.

        Returns:
            node: узел дерева.
        """
        available_feature_names = available_feature_names.copy()

        # выбор лучшего признака для разбиения
        impurity = self.__impurity(Y)

        best_feature = None
        best_gain = self.__min_impurity_decrease
        for feature_name in available_feature_names:
            current_gain = self.__information_gain(X, Y, feature_name, impurity)
            if isinstance(best_gain, float) and isinstance(current_gain, float):
                if best_gain < current_gain:
                    best_gain = current_gain
                    best_feature = feature_name
            elif isinstance(best_gain, tuple) and isinstance(current_gain, float):
                if best_gain[0] < current_gain:
                    best_gain = current_gain
                    best_feature = feature_name
            elif isinstance(best_gain, float) and isinstance(current_gain, tuple):
                if best_gain < current_gain[0]:
                    best_gain = current_gain
                    best_feature = feature_name
            elif isinstance(best_gain, tuple) and isinstance(current_gain, tuple):
                if best_gain[0] < current_gain[0]:
                    best_gain = current_gain
                    best_feature = feature_name

        samples = X.shape[0]
        distribution = []
        label = None
        max_samples_per_class = -1
        for class_name in self.class_names:
            samples_per_class = (Y == class_name).sum()
            distribution.append(samples_per_class)
            if max_samples_per_class < samples_per_class:
                max_samples_per_class = samples_per_class
                label = class_name

        if best_feature:
            if isinstance(best_gain, float):
                self.__feature_importances_[best_feature] += (samples/self.__total_samples) * best_gain
            elif isinstance(best_gain, tuple):
                self.__feature_importances_[best_feature] += (samples/self.__total_samples) * best_gain[0]

        childs = []
        if best_feature:
            # удаление категориальных признаков
            if best_feature in self.__categorical_feature_names:
                available_feature_names.remove(best_feature)
            # добавление открывшихся признаков
            if special_cases:
                if best_feature in special_cases.keys():
                    if isinstance(special_cases[best_feature], str):
                        available_feature_names.append(special_cases[best_feature])
                    elif isinstance(special_cases[best_feature], list):
                        available_feature_names.extend(special_cases[best_feature])
                    special_cases.pop(best_feature)
            # рекурсивное создание потомков
            num_samples = X.shape[0]
            if num_samples >= self.__min_samples_split:
                xs, ys, feature_values = self.__split(X, Y, best_feature, best_gain)

                for x, y, fv in zip(xs, ys, feature_values):
                    childs.append(
                        self.__generate_node(x, y, fv, available_feature_names, special_cases)
                    )

        node = Node(best_feature, feature_value, impurity, samples, distribution, label, childs)

        return node

    def __impurity(self, Y):
        """Считает загрязнённость для множества.

        Args:
            Y (pd.Series): c метками для множества.

        Returns:
            impurity: загрязнённость множества.
        """
        impurity = None
        if self.__criterion == 'entropy':
            impurity = self.__entropy(Y)

        return impurity

    @staticmethod
    def __entropy(Y):
        """Считает энтропию в множестве.

        Args:
            Y (pd.Series): c метками для множества.

        Returns:
            entropy: энтропия множества.
        """
        n = Y.shape[0]  # количество точек в множестве

        entropy = 0
        for label in set(Y.tolist()):  # перебор по классам
            m_i = (Y == label).sum()
            if m_i != 0:
                entropy -= (m_i/n) * math.log2(m_i/n)

        return entropy

    def __information_gain(self, X, Y, feature_name, impurity):
        """Возвращает прирост информативности для разделения по признаку.

        Формула в LaTeX:
        Gain(A, Q) = H(A, S) -\sum\limits^q_{i=1} \frac{|A_i|}{|A|} H(A_i, S),
        где
        A - множество точек данных;
        Q - метка данных;
        q - множество значений метки, т.е. классы;
        S - признак;
        A_i - множество элементов A, на которых Q имеет значение i.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_name: название признака, по которому будет проходить разделение.
            impurity: загрязнённость до разделения множества по признаку.

        Returns:
            information_gain: прирост информативности.
        """
        information_gain = None
        if feature_name in self.__categorical_feature_names:
            information_gain = self.__categorical_information_gain(X, Y, feature_name, impurity)
        elif feature_name in self.__numerical_feature_names:
            information_gain = self.__numerical_information_gain(X, Y, feature_name, impurity)

        return information_gain

    def __categorical_information_gain(self, X, Y, feature_name, impurity):
        """Считает прирост информации для разделения по категориальному признаку."""
        A = X.shape[0]
        second_term = 0
        for feature_value in set(X[feature_name].tolist()):
            A_i = (X[feature_name] == feature_value).sum()
            y_i = Y[X[feature_name] == feature_value]
            second_term += (A_i/A) * self.__impurity(y_i)

        categorical_information_gain = impurity - second_term

        return categorical_information_gain

    def __numerical_information_gain(self, X, Y, feature_name, impurity):
        """Считает прирост информации для разделения по численному признаку."""
        A = X.shape[0]

        best_gain = 0
        best_threshold = None
        for threshold in range(int(X[feature_name].max())):
            A_less = (X[feature_name] < threshold).sum()
            y_less = Y[X[feature_name] < threshold]
            A_more = (X[feature_name] >= threshold).sum()
            y_more = Y[X[feature_name] >= threshold]

            numerical_information_gain = (
                    impurity -
                    (A_less/A) * self.__impurity(y_less) -
                    (A_more/A) * self.__impurity(y_more)
            )

            if best_gain is None or best_gain < numerical_information_gain:
                best_gain = numerical_information_gain
                best_threshold = threshold

        return best_gain, best_threshold

    def __split(self, X, Y, feature_name, threshold):
        """Разделяет множество по признаку.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_name: признак, по которому будет происходить разделение.
            threshold: порог разбиения для численного признака.

        Returns:
            xs: список с подмножествами точек данных.
            ys: список с подмножествами соответствующих меток.
            feature_values: список со значениями признака, по которому расщепляется множество.
        """
        xs, ys, feature_values = None, None, None
        if feature_name in self.__categorical_feature_names:
            xs, ys, feature_values = self.__categorical_split(X, Y, feature_name)
        elif feature_name in self.__numerical_feature_names:
            xs, ys, feature_values = self.__numerical_split(X, Y, feature_name, threshold[1])

        return xs, ys, feature_values

    @staticmethod
    def __categorical_split(X, Y, feature_name):
        """Расщепляет множество согласно категориальному признаку."""
        xs, ys, feature_values = [], [], []
        for feature_value in set(X[feature_name].tolist()):
            xs.append(X[X[feature_name] == feature_value])
            ys.append(Y[X[feature_name] == feature_value])
            feature_values.append(feature_value)

        return xs, ys, feature_values

    @staticmethod
    def __numerical_split(X, Y, feature_name, threshold):
        """Расщепляет множество согласно численному признаку."""
        x_less = X[X[feature_name] < threshold]
        x_more = X[X[feature_name] >= threshold]
        xs = [x_less, x_more]
        y_less = Y[X[feature_name] < threshold]
        y_more = Y[X[feature_name] >= threshold]
        ys = [y_less, y_more]
        feature_values = [f'< {threshold}', f'>= {threshold}']

        return xs, ys, feature_values

    def render(
            self,
            *,
            rounded=False,
            show_impurity=False,
            show_num_samples=False,
            show_distribution=False,
            show_label=False,
            **kwargs):
        """Визуализирует дерево решений.

        Если указаны именованные параметры, сохраняет визуализацию в виде файла(ов).

        Args:
            rounded: скруглять ли углы у узлов (они форме прямоугольника).
            show_impurity: показывать ли загрязнённость узла.
            show_num_samples: показывать ли количество точек в узле.
            show_distribution: показывать ли распределение точек по классам.
            show_label: показывать ли класс, к которому относится узел.

        Returns:
            graph: объект класса Digraph, содержащий описание графовой структуры дерева для
              визуализации.
        """
        if self.__graph is None:
            self.__create_graph(
                rounded, show_impurity, show_num_samples, show_distribution, show_label)
        if kwargs:
            self.__graph.render(**kwargs)

        return self.__graph

    def __create_graph(
            self, rounded, show_impurity, show_num_samples, show_distribution, show_label):
        """Создаёт объект класса Digraph, содержащий описание графовой структуры дерева для
        визуализации."""
        node_attr = {'shape': 'box'}
        if rounded:
            node_attr['style'] = 'rounded'
        self.__graph = Digraph(name='дерево решений', node_attr=node_attr)
        self.__add_node(
            self.tree, None, show_impurity, show_num_samples, show_distribution, show_label
        )

    @counter
    def __add_node(
            self,
            node,
            parent_name,
            show_impurity,
            show_num_samples,
            show_distribution,
            show_label
    ):
        """Рекурсивно добавляет описание узла и его связь с родительским узлом (если имеется)."""
        node_name = f'node{self.__add_node.count}'
        node_content = ''
        if node.split_feature:
            node_content += f'{node.split_feature}\n'
        if show_impurity:
            node_content += f'{self.__criterion} = {node.impurity:2.2}\n'
        if show_num_samples:
            node_content += f'samples = {node.samples}\n'
        if show_distribution:
            node_content += f'distribution: {node.distribution}\n'
        if show_label:
            node_content += f'label = {node.label}'

        self.__graph.node(name=node_name, label=node_content)
        if parent_name:
            self.__graph.edge(parent_name, node_name, label=f'{node.feature_value}')

        for child in node.childs:
            self.__add_node(
                child, node_name, show_impurity, show_num_samples, show_distribution, show_label
            )


class Node:
    """Узел дерева решений."""
    def __init__(
            self,
            split_feature,
            feature_value,
            impurity,
            samples,
            distribution,
            label,
            childs
    ):
        self.split_feature = split_feature
        self.feature_value = feature_value
        self.impurity = impurity
        self.samples = samples
        self.distribution = distribution
        self.label = label
        self.childs = childs
