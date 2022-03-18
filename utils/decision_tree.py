"""Кастомная реализация дерева решений, которая может работать с категориальными и численными
признаками.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

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
    """Дерево решений.

    Attributes:
        feature_names: список всех признаков, находившихся в обучающих данных.
        class_names: отсортированный список классов.
        categorical_feature_names: список всех категориальных признаков, находившихся в обучающих
          данных.
        numerical_feature_names: список всех численных признаков, находившихся в обучающих данных.
        feature_importances: словарь, в котором ключ представляет собой название признака, а
          значение - его нормализованная значимость.
    """
    def __init__(
            self,
            *,
            max_depth: Optional[int] = None,
            criterion: Optional[str] = 'entropy',
            min_samples_split: Optional[int] = 2,
            min_samples_leaf: Optional[int] = 1,
            min_impurity_decrease: Optional[float] = 0.05,
    ) -> None:
        if max_depth is not None and not isinstance(max_depth, int):
            raise ValueError('max_depth должен представлять собой int.')

        if criterion not in ['entropy', 'gini']:
            raise ValueError('Для criterion доступны значения "entropy" и "gini".')

        if not isinstance(min_samples_split, int) or min_samples_split <= 1:
            raise ValueError(
                'min_samples_split должен представлять собой int и быть строго больше 1.'
            )

        if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
            raise ValueError(
                'min_samples_leaf должен представлять собой int и быть строго больше 0.'
            )

        if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0:
            raise ValueError(
                'min_impurity_decrease должен представлять собой float и быть неотрицательным.'
            )

        if min_samples_split < 2 * min_samples_leaf:
            raise ValueError(
                'min_samples_split должен быть строго в 2 раза больше min_samples_leaf.'
            )
        self.__max_depth = max_depth
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

        self.__tree = self.__generate_node(X, Y, None, 1, available_feature_names, special_cases)

    @counter
    def __generate_node(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_value: str,
            depth: int,
            available_feature_names: List[str],
            special_cases: Optional[Dict[str, Union[str, Dict]]] = None,
    ) -> Node:
        """Рекурсивная функция создания узлов дерева.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_value: значение признака, по которому этот узел был сформирован.
            depth: глубина узла.
            available_feature_names: список доступных признаков для разбиения входного множества.
            special_cases: словарь {признак, который должен быть первым: признак, который может быть
              после}.

        Returns:
            node: узел дерева.
        """
        impurity = self.__impurity(Y)
        samples = X.shape[0]
        distribution = self.__distribution(Y)
        label = self.__label(Y)

        childs = []
        feature = None
        if samples >= self.__min_samples_split and \
                (not self.__max_depth or (depth <= self.__max_depth)):
            feature, xs, ys, feature_values, inf_gain = self.__split(X, Y, available_feature_names)

            if feature:
                available_feature_names = available_feature_names.copy()
                special_cases = special_cases.copy()

                self.__feature_importances[feature] += \
                    (samples/self.__total_samples) * inf_gain

                # удаление категориальных признаков
                if feature in self.__categorical_feature_names:
                    available_feature_names.remove(feature)
                # добавление открывшихся признаков
                if special_cases:
                    if feature in special_cases.keys():
                        if isinstance(special_cases[feature], str):
                            available_feature_names.append(special_cases[feature])
                        elif isinstance(special_cases[feature], list):
                            available_feature_names.extend(special_cases[feature])
                        else:
                            assert False, 'пришли сюда'
                        special_cases.pop(feature)

                # рекурсивное создание потомков
                for x, y, fv in zip(xs, ys, feature_values):
                    child = self.__generate_node(
                        x, y, fv, depth + 1, available_feature_names, special_cases
                    )
                    childs.append(child)

        assert label is not None, 'label is None'

        node = Node(feature, feature_value, impurity, samples, distribution, label, childs)

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
        """Считает загрязнённость для множества."""
        impurity = None
        if self.__criterion == 'entropy':
            impurity = self.__entropy(Y)
        elif self.__criterion == 'gini':
            impurity = self.__gini(Y)

        return impurity

    def __entropy(self, Y: pd.Series) -> float:
        """Считает энтропию в множестве."""
        n = Y.shape[0]  # количество точек в множестве

        entropy = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (Y == label).sum()
            if m_i != 0:
                entropy -= (m_i/n) * math.log2(m_i/n)

        return entropy

    def __gini(self, Y: pd.Series) -> float:
        """Считает коэффициент Джини в множестве."""
        n = Y.shape[0]  # количество точек в множестве

        gini = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (Y == label).sum()
            p_i = m_i/n
            gini += p_i * (1 - p_i)

        return gini

    def __split(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            available_feature_names: List[str],
    ) -> Tuple[str, List[pd.DataFrame], List[pd.Series], List[str], float]:
        """Разделяет входное множество наилучшим образом.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            available_feature_names: список доступных признаков для разбиения входного множества.

        Returns:
            best_feature_name: признак, по которому лучше всего разбивать входное множество.
            best_xs: список DataFrame'ов с точками данных дочерних подмножеств.
            best_ys: список Series с соответствующими метками дочерних подмножеств.
            best_feature_values: значения признаков, соответствующие дочерним подмножествам.
            best_inf_gain: прирост информативности после разбиения.
        """
        best_feature_name = None
        best_xs = []
        best_ys = []
        best_feature_values = []
        best_inf_gain = 0
        for feature_name in available_feature_names:
            if feature_name in self.__categorical_feature_names:
                inf_gain, xs, ys, feature_values = self.__categorical_split(X, Y, feature_name)
            elif feature_name in self.__numerical_feature_names:
                inf_gain, xs, ys, feature_values = self.__numerical_split(X, Y, feature_name)
            else:
                assert False, 'пришли сюда'

            if inf_gain >= self.__min_impurity_decrease and inf_gain > best_inf_gain:
                best_inf_gain = inf_gain
                best_feature_name = feature_name
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        return best_feature_name, best_xs, best_ys, best_feature_values, best_inf_gain

    def __categorical_split(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series], List[str]]:
        """Разделяет входное множество по категориальному признаку.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            inf_gain: прирост информативности при разделении.
            xs: список DataFrame'ов с точками данных дочерних подмножеств.
            ys: список Series с соответствующими метками дочерних подмножеств.
            feature_values: значения признаков, соответствующие дочерним подмножествам.
        """
        # список со значениями признака
        partition = sorted(list(set(X[feature_name].tolist())))

        xs = []
        ys = []
        feature_values = []
        for feature_value in partition:
            if (X[feature_name] == feature_value).sum() < self.__min_samples_leaf:
                return 0, [], [], []
            else:
                xs.append(X[X[feature_name] == feature_value])
                ys.append(Y[X[feature_name] == feature_value])
                feature_values.append(feature_value)

        inf_gain = self.__information_gain(Y, ys)

        return inf_gain, xs, ys, feature_values

    def __numerical_split(
            self,
            X: pd.DataFrame,
            Y: pd.Series,
            feature_name: str,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series], List[str]]:
        """Разделяет входное множество по численному признаку, выбирая наилучший порог.

        Args:
            X: DataFrame с точками данных.
            Y: Series с соответствующими метками.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            best_inf_gain: прирост информативности при разделении.
            best_xs: список DataFrame'ов с точками данных дочерних подмножеств.
            best_ys: список Series с соответствующими метками дочерних подмножеств.
            best_feature_values: значения признаков, соответствующие дочерним подмножествам.
        """
        best_inf_gain = 0
        best_xs = []
        best_ys = []
        best_feature_values = []
        for threshold in range(int(X[feature_name].min()) + 1, int(X[feature_name].max())):
            x_less = X[X[feature_name] < threshold]
            y_less = Y[X[feature_name] < threshold]
            A_less = y_less.shape[0]

            x_more = X[X[feature_name] >= threshold]
            y_more = Y[X[feature_name] >= threshold]
            A_more = y_more.shape[0]

            xs = [x_less, x_more]
            ys = [y_less, y_more]
            feature_values = [f'< {threshold}', f'>= {threshold}']

            if A_less < self.__min_samples_leaf or A_more < self.__min_samples_leaf:
                continue

            inf_gain = self.__information_gain(Y, ys)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        return best_inf_gain, best_xs, best_ys, best_feature_values

    def __information_gain(
            self,
            Y: pd.Series,
            ys: List[pd.Series],
    ) -> float:
        """Считает прирост информативности.

        Формула в LaTeX:
        Gain(A, Q) = H(A, S) -\sum\limits^q_{i=1} \frac{|A_i|}{|A|} H(A_i, S),
        где
        H - функция энтропии;
        A - множество точек данных;
        Q - метка данных;
        q - множество значений метки, т.е. классы;
        S - признак;
        A_i - множество элементов A, на которых Q имеет значение i.

        Args:
            Y: Series с метками родительского множества.
            ys: список Series с метками дочерних подмножеств.

        Returns:
            inf_gain: прирост информативности.
        """
        A = Y.shape[0]

        second_term = 0
        for y_i in ys:
            A_i = y_i.shape[0]
            second_term += (A_i/A) * self.__impurity(y_i)

        inf_gain = self.__impurity(Y) - second_term

        return inf_gain

    def get_params(
            self,
            deep: Optional[bool] = True,
    ) -> Dict:
        """Возвращает параметры этого классификатора."""
        params = {
            'max_depth': self.__max_depth,
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
        if node.feature is None:
            Y = node.label
            assert Y is not None, 'label оказался None'
        elif node.feature in self.__categorical_feature_names:
            # ищем ту ветвь, по которой нужно идти
            for child in node.childs:
                if child.feature_value == point[node.feature]:
                    Y = self.__predict(child, point)
                    break
            else:
                # если такой ветви нет
                if Y is None:
                    Y = node.label
        elif node.feature in self.__numerical_feature_names:
            # ищем ту ветвь, по которой нужно идти
            threshold = float(node.childs[0].feature_value[2:])
            if point[node.feature] < threshold:
                Y = self.__predict(node.childs[0], point)
            elif point[node.feature] >= threshold:
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
        if node.feature:
            node_content += f'{node.feature}\n'
        if show_impurity:
            node_content += f'{self.__criterion} = {node.impurity:.3f}\n'
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
            feature,
            feature_value,
            impurity,
            samples,
            distribution,
            label,
            childs,
    ):
        self.feature = feature
        self.feature_value = feature_value
        self.impurity = impurity
        self.samples = samples
        self.distribution = distribution
        self.label = label
        self.childs = childs
