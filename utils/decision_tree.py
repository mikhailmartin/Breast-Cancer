"""Кастомная реализация дерева решений, которая может работать с категориальными и численными
признаками.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import functools
import math

from graphviz import Digraph
import pandas as pd


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
            min_impurity_decrease=0.05,
    ) -> None:
        self.__criterion = criterion
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_impurity_decrease = min_impurity_decrease

        self.__feature_names = None
        self.__class_names = None
        self.__categorical_feature_names = None
        self.__numerical_feature_names = None
        self.__tree = None
        self.__graph = None
        self.__feature_importances = None
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
    def feature_importances(self):
        total = 0
        for importance in self.__feature_importances.values():
            total += importance
        for feature in self.__feature_importances.keys():
            self.__feature_importances[feature] /= total

        return self.__feature_importances

    def fit(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            categorical_feature_names: List[str],
            numerical_feature_names: List[str],
            *,
            special_cases: Optional[Dict[str, Union[str, Dict]]] = None,
    ) -> None:
        """Обучает дерево решений.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            categorical_feature_names: список категориальных признаков.
            numerical_feature_names: список численных признаков.
            special_cases: словарь {признак, который должен быть первым: признак или список
              признаков, которые могут быть после}.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X должен представлять собой pd.DataFrame.')

        if not isinstance(Y, pd.Series):
            raise ValueError('Y должен представлять собой pd.Series.')

        if X.shape[0] != Y.shape[0]:
            raise ValueError('X и Y должны быть одной длины.')

        if not isinstance(categorical_feature_names, list):
            raise ValueError('categorical_feature_names должен представлять собой список строк.')
        for elem in categorical_feature_names:
            if not isinstance(elem, str):
                raise ValueError(
                    'categorical_feature_names должен представлять собой список строк.'
                    )

        if not isinstance(numerical_feature_names, list):
            raise ValueError('numerical_feature_names должен представлять собой список строк.')
        for elem in numerical_feature_names:
            if not isinstance(elem, str):
                raise ValueError('numerical_feature_names должен представлять собой список строк.')

        if special_cases:
            if not isinstance(special_cases, dict):
                raise ValueError(
                    'special_cases должен представлять собой словарь, в котором ключи - строки, а '
                    'значения - либо строки, либо списки строк.'
                )
            for key in special_cases.keys():
                if not isinstance(key, str):
                    raise ValueError(
                        'special_cases должен представлять собой словарь, в котором ключи - '
                        'строки, а значения - либо строки, либо списки строк.'
                    )
            for value in special_cases.values():
                if not isinstance(value, (str, list)):
                    raise ValueError(
                        'special_cases должен представлять собой словарь, в котором ключи - '
                        'строки, а значения - либо строки, либо списки строк.'
                    )
                if isinstance(value, list):
                    for elem in value:
                        if not isinstance(elem, str):
                            raise ValueError(
                                'special_cases должен представлять собой словарь, в котором '
                                'ключи - строки, а значения - либо строки, либо списки строк.'
                            )

        for feature_name in categorical_feature_names:
            if feature_name not in X.columns:
                raise ValueError(
                    f'categorical_feature_names содержит признак {feature_name}, которого нет в '
                    'обучающих данных.'
                )
        for feature_name in numerical_feature_names:
            if feature_name not in X.columns:
                raise ValueError(
                    f'numerical_feature_names содержит признак {feature_name}, которого нет в '
                    'обучающих данных.'
                )
        for feature_name in X.columns:
            if feature_name not in categorical_feature_names + numerical_feature_names:
                raise ValueError(
                    f'Обучающие данные содержат признак {feature_name}, который не определён ни в '
                    f'categorical_feature_names, ни в numerical_feature_names.'
                )

        self.__feature_names = list(X.columns)
        self.__class_names = sorted(list(set(Y.tolist())))
        self.__categorical_feature_names = categorical_feature_names
        self.__numerical_feature_names = numerical_feature_names

        self.__total_samples = X.shape[0]
        self.__feature_importances = dict.fromkeys(self.__feature_names, 0)

        available_feature_names = self.__feature_names.copy()
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
    def __generate_node(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_value: str,
            available_feature_names: List[str],
            special_cases: Optional[Dict[str, Union[str, Dict]]] = None,
    ) -> Node:
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
        special_cases = special_cases.copy()

        impurity = self.__impurity(Y)

        split_feature, split_gain, threshold = self.__choose_split_feature(
            X, Y, available_feature_names, impurity
        )

        samples = X.shape[0]
        distribution = self.__distribution(Y)
        label = self.__label(Y)

        if split_feature:
            self.__feature_importances[split_feature] += (samples/self.__total_samples) * split_gain

        childs = []
        # рекурсивное создание потомков
        if split_feature and (samples >= self.__min_samples_split):
            xs, ys, feature_values = self.__split(X, Y, split_feature, threshold)

            # удаление категориальных признаков
            if split_feature in self.__categorical_feature_names:
                available_feature_names.remove(split_feature)
            # добавление открывшихся признаков
            if special_cases:
                if split_feature in special_cases.keys():
                    if isinstance(special_cases[split_feature], str):
                        available_feature_names.append(special_cases[split_feature])
                    elif isinstance(special_cases[split_feature], list):
                        available_feature_names.extend(special_cases[split_feature])
                    else:
                        assert False, 'пришли сюда'
                    special_cases.pop(split_feature)

            for x, y, fv in zip(xs, ys, feature_values):
                childs.append(
                    self.__generate_node(x, y, fv, available_feature_names, special_cases)
                )

        assert label is not None, 'label is None'

        node = Node(split_feature, feature_value, impurity, samples, distribution, label, childs)

        return node

    def __distribution(self, Y: pd.Series) -> List[int]:
        """Подсчитывает распределение точек данных по классам."""
        distribution = [(Y == class_name).sum() for class_name in self.__class_names]

        return distribution

    def __label(self, Y: pd.Series) -> str:
        """Выбирает метку для узла дерева."""
        label = None
        max_samples_per_class = -1
        for class_name in self.__class_names:
            samples_per_class = (Y == class_name).sum()
            if max_samples_per_class < samples_per_class:
                max_samples_per_class = samples_per_class
                label = class_name

        return label

    def __impurity(self, Y: pd.Series) -> float:
        """Считает загрязнённость для множества.

        Args:
            Y: c метками для множества.

        Returns:
            impurity: загрязнённость множества.
        """
        impurity = None
        if self.__criterion == 'entropy':
            impurity = self.__entropy(Y)
        elif self.__criterion == 'gini':
            impurity = self.__gini(Y)

        return impurity

    def __entropy(self, Y: pd.Series) -> float:
        """Считает энтропию в множестве.

        Args:
            Y: c метками для множества.

        Returns:
            entropy: энтропия множества.
        """
        n = Y.shape[0]  # количество точек в множестве

        entropy = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (Y == label).sum()
            if m_i != 0:
                entropy -= (m_i/n) * math.log2(m_i/n)

        return entropy

    def __gini(self, Y: pd.Series) -> float:
        """Считает коэффициент Джини в множестве.

        Args:
            Y: c метками для множества.

        Returns:
            gini: коэффициент Джини.
        """
        n = Y.shape[0]  # количество точек в множестве

        gini = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (Y == label).sum()
            p_i = m_i/n
            gini += p_i * (1 - p_i)

        return gini

    def __choose_split_feature(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            available_feature_names: List[str],
            impurity: float,
    ) -> Tuple[Any, float, Any]:
        """Выбирает лучший признак для разбиения."""
        best_feature = None
        best_gain = 0
        threshold = None

        for feature_name in available_feature_names:
            current_gain, current_threshold = self.__information_gain(X, Y, feature_name, impurity)
            if current_gain >= self.__min_impurity_decrease and current_gain > best_gain:
                best_feature = feature_name
                best_gain = current_gain
                threshold = current_threshold

        return best_feature, best_gain, threshold

    def __information_gain(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
            impurity: float,
    ) -> Tuple[float, Any]:
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
            inf_gain: прирост информативности.
            threshold: порог принятия решения для численного признака.
        """
        if feature_name in self.__categorical_feature_names:
            inf_gain, threshold = self.__categorical_information_gain(X, Y, feature_name, impurity)
        elif feature_name in self.__numerical_feature_names:
            inf_gain, threshold = self.__numerical_information_gain(X, Y, feature_name, impurity)
        else:
            assert False, 'пришли сюда'

        return inf_gain, threshold

    def __categorical_information_gain(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
            impurity: float,
    ) -> Tuple[float, None]:
        """Считает прирост информации для разделения по категориальному признаку."""
        A = X.shape[0]
        second_term = 0
        for feature_value in set(X[feature_name].tolist()):
            A_i = (X[feature_name] == feature_value).sum()
            if A_i < self.__min_samples_leaf:
                return 0, None
            y_i = Y[X[feature_name] == feature_value]
            second_term += (A_i/A) * self.__impurity(y_i)

        categorical_information_gain = impurity - second_term

        return categorical_information_gain, None

    def __numerical_information_gain(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
            impurity: float,
    ) -> Tuple[float, int]:
        """Считает прирост информации для разделения по численному признаку."""
        A = X.shape[0]

        best_information_gain, best_threshold = 0, None
        for threshold in range(int(X[feature_name].max())):
            A_less = (X[feature_name] < threshold).sum()
            y_less = Y[X[feature_name] < threshold]
            A_more = (X[feature_name] >= threshold).sum()
            y_more = Y[X[feature_name] >= threshold]
            # проверка на пустое дочернее множество
            if A_less < self.__min_samples_leaf or A_more < self.__min_samples_leaf:
                continue

            current_information_gain = (
                    impurity -
                    (A_less/A) * self.__impurity(y_less) -
                    (A_more/A) * self.__impurity(y_more)
            )

            if best_information_gain < current_information_gain:
                best_information_gain = current_information_gain
                best_threshold = threshold

        return best_information_gain, best_threshold

    def __split(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
            threshold: int,
    ) -> Tuple[List, List, List]:
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
            xs, ys, feature_values = self.__numerical_split(X, Y, feature_name, threshold)

        return xs, ys, feature_values

    @staticmethod
    def __categorical_split(
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
    ) -> Tuple[List, List, List]:
        """Расщепляет множество согласно категориальному признаку."""
        xs, ys, feature_values = [], [], []
        for feature_value in sorted(list(set(X[feature_name].tolist()))):
            if (X[feature_name] == feature_value).sum():
                xs.append(X[X[feature_name] == feature_value])
                ys.append(Y[X[feature_name] == feature_value])
                feature_values.append(feature_value)

        return xs, ys, feature_values

    @staticmethod
    def __numerical_split(
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
            threshold,
    ) -> Tuple[List, List, List]:
        """Расщепляет множество согласно численному признаку."""
        x_less = X[X[feature_name] < threshold]
        x_more = X[X[feature_name] >= threshold]
        xs = [x_less, x_more]
        y_less = Y[X[feature_name] < threshold]
        y_more = Y[X[feature_name] >= threshold]
        ys = [y_less, y_more]
        feature_values = [f'< {threshold}', f'>= {threshold}']

        return xs, ys, feature_values

    def get_params(self, deep: Optional[bool] = True) -> Dict:
        """Возвращает параметры этого классификатора."""
        params = {
            'min_samples_split': self.__min_samples_split,
            'min_samples_leaf': self.__min_samples_leaf,
            'min_impurity_decrease': self.__min_impurity_decrease,
        }

        return params

    def set_params(self, **params):
        """Задаёт параметры этому классификатору."""
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f'Invalid parameter {param} for estimator {self}. '
                    'Check the list of available parameters '
                    'with `estimator.get_params().keys()`.'
                )

            setattr(self, param, value)
            valid_params[param] = value

        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series]) -> Union[List[str], str]:
        """Предсказывает метки классов для точек данных в X."""
        if isinstance(X, pd.DataFrame):
            Y = [self.predict(point) for _, point in X.iterrows()]
        elif isinstance(X, pd.Series):
            Y = self.__predict(self.__tree, X)
        else:
            raise ValueError('X должен представлять собой pd.DataFrame или pd.Series.')

        assert Y is not None, 'предсказывает None'

        return Y

    def __predict(self, node: Node, point: pd.Series) -> str:
        """Предсказывает метку класса для точки данных."""
        Y = None
        # если мы дошли до листа
        if node.split_feature is None:
            Y = node.label
            assert Y is not None, 'label оказался None'
        elif node.split_feature in self.__categorical_feature_names:
            # ищем ту ветвь, по которой нужно идти
            for child in node.childs:
                if child.feature_value == point[node.split_feature]:
                    Y = self.__predict(child, point)
                    break
            else:
                # если такой ветви нет
                if Y is None:
                    Y = node.label
        elif node.split_feature in self.__numerical_feature_names:
            # ищем ту ветвь, по которой нужно идти
            threshold = float(node.childs[0].feature_value[2:])
            if point[node.split_feature] < threshold:
                Y = self.__predict(node.childs[0], point)
            elif point[node.split_feature] >= threshold:
                Y = self.__predict(node.childs[1], point)
            else:
                assert False, 'пришли сюда'
        else:
            assert False, ('node.split_feature и не None, и не в categorical_feature_names и не в'
                           'numerical_feature_names')

        assert Y is not None, 'Y is None'

        return Y

    def render(
            self,
            *,
            rounded: Optional[bool] = False,
            show_impurity: Optional[bool] = False,
            show_num_samples: Optional[bool] = False,
            show_distribution: Optional[bool] = False,
            show_label: Optional[bool] = False,
            **kwargs,
    ) -> Digraph:
        """Визуализирует дерево решений.

        Если указаны именованные параметры, сохраняет визуализацию в виде файла(ов).

        Args:
            rounded: скруглять ли углы у узлов (они форме прямоугольника).
            show_impurity: показывать ли загрязнённость узла.
            show_num_samples: показывать ли количество точек в узле.
            show_distribution: показывать ли распределение точек по классам.
            show_label: показывать ли класс, к которому относится узел.
            **kwargs: аргументы для graphviz.Digraph.render.

        Returns:
            graph: объект класса Digraph, содержащий описание графовой структуры дерева для
              визуализации.
        """
        if self.__graph is None:
            self.__create_graph(
                rounded, show_impurity, show_num_samples, show_distribution, show_label
            )
        if kwargs:
            self.__graph.render(**kwargs)

        return self.__graph

    def __create_graph(
            self,
            rounded: bool,
            show_impurity: bool,
            show_num_samples: bool,
            show_distribution: bool,
            show_label: bool,
    ) -> None:
        """Создаёт объект класса Digraph, содержащий описание графовой структуры дерева для
        визуализации."""
        node_attr = {'shape': 'box'}
        if rounded:
            node_attr['style'] = 'rounded'
        self.__graph = Digraph(name='дерево решений', node_attr=node_attr)
        self.__add_node(
            self.__tree, None, show_impurity, show_num_samples, show_distribution, show_label
        )

    @counter
    def __add_node(
            self,
            node: Node,
            parent_name: str,
            show_impurity: bool,
            show_num_samples: bool,
            show_distribution: bool,
            show_label: bool,
    ) -> None:
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

    def score(self, X: pd.DataFrame, Y: pd.Series) -> float:
        """Возвращает точность по заданным тестовым данным и меткам."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X должен представлять собой pd.DataFrame.')

        if not isinstance(Y, pd.Series):
            raise ValueError('Y должен представлять собой pd.Series.')

        if X.shape[0] != Y.shape[0]:
            raise ValueError('X и Y должны быть одной длины.')

        from sklearn.metrics import accuracy_score

        score = accuracy_score(Y, self.predict(X))

        return score


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
            childs,
    ):
        self.split_feature = split_feature
        self.feature_value = feature_value
        self.impurity = impurity
        self.samples = samples
        self.distribution = distribution
        self.label = label
        self.childs = childs
