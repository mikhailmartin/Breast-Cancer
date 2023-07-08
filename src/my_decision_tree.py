"""Кастомная реализация дерева решений, которая может работать с категориальными и численными
признаками.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import functools
import math

from graphviz import Digraph
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def counter(function):
    """Декоратор-счётчик."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return function(*args, **kwargs)
    wrapper.count = 0
    return wrapper


def categorical_partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in categorical_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def rank_partition(collection):
    for i in range(1, len(collection)):
        yield collection[:i], collection[i:]


class MyDecisionTreeClassifier:
    """
    Дерево решений.

    Attributes:
        feature_names: список всех признаков, находившихся в обучающих данных.
        class_names: отсортированный список классов.
        categorical_feature_names: список всех категориальных признаков, находившихся в обучающих
          данных.
        rank_feature_names: словарь, в котором ключ представляет собой название рангового признака,
          а ключ - упорядоченный список его значений.
        numerical_feature_names: список всех численных признаков, находившихся в обучающих данных.
        feature_importances: словарь, в котором ключ представляет собой название признака, а
          значение - его нормализованная значимость.
    """
    def __init__(
            self,
            *,
            max_depth: Optional[int] = None,
            criterion: Optional[str] = 'gini',
            min_samples_split: Optional[int] = 2,
            min_samples_leaf: Optional[int] = 1,
            min_impurity_decrease: Optional[float] = .0,
            max_childs: Optional[int] = None,
    ) -> None:
        if max_depth is not None and not isinstance(max_depth, int):
            raise ValueError('`max_depth` должен представлять собой int.')

        if criterion not in ['entropy', 'gini']:
            raise ValueError('Для `criterion` доступны значения "entropy" и "gini".')

        if not isinstance(min_samples_split, int) or min_samples_split <= 1:
            raise ValueError(
                '`min_samples_split` должен представлять собой int и быть строго больше 1.'
            )

        if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
            raise ValueError(
                '`min_samples_leaf` должен представлять собой int и быть строго больше 0.'
            )

        if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0:
            raise ValueError(
                '`min_impurity_decrease` должен представлять собой float и быть неотрицательным.'
            )

        if min_samples_split < 2 * min_samples_leaf:
            raise ValueError(
                '`min_samples_split` должен быть строго в 2 раза больше min_samples_leaf.'
            )

        if max_childs is not None and not isinstance(max_childs, int) or \
                isinstance(max_childs, int) and max_childs < 2:
            raise ValueError(
                '`max_childs` должен представлять собой int и быть строго больше 1.'
            )
        self.__max_depth = max_depth
        self.__criterion = criterion
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_impurity_decrease = min_impurity_decrease
        self.__max_childs = max_childs

        self.__feature_names = None
        self.__class_names = None
        self.__cat_feature_names = {}
        self.__rank_feature_names = {}
        self.__num_feature_names = []
        self.__tree = None
        self.__graph = None
        self.__feature_importances = None
        self.__total_samples_num = None

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def class_names(self):
        return self.__class_names

    @property
    def categorical_feature_names(self) -> Dict[str, List[str]]:
        return self.__cat_feature_names

    @property
    def rank_feature_names(self) -> Dict[str, List]:
        return self.__rank_feature_names

    @property
    def numerical_feature_names(self) -> List[str]:
        return self.__num_feature_names

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
            y: pd.Series,
            *,
            categorical_feature_names: Optional[Dict[str, List[str]]] = None,
            rank_feature_names: Optional[Dict[str, List]] = None,
            numerical_feature_names: Optional[List] = None,
            special_cases: Optional[Dict[str, str | Dict]] = None,
    ) -> None:
        """
        Обучает дерево решений.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            categorical_feature_names: словарь, в котором ключ представляет собой название
              категориального признака, а значение - список возможных его значений.
            rank_feature_names: словарь, в котором ключ представляет собой название рангового
              признака, а значение - упорядоченный список его значений.
            numerical_feature_names: список численных признаков.
            special_cases: словарь {признак, который должен быть первым: признак или список
              признаков, которые могут быть после}.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X должен представлять собой pd.DataFrame.')

        if not isinstance(y, pd.Series):
            raise ValueError('y должен представлять собой pd.Series.')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X и y должны быть одной длины.')

        if not any((categorical_feature_names, rank_feature_names, numerical_feature_names)):
            raise ValueError(
                'Признаки должны быть отнесены хотя бы к одной из возможных групп '
                '(categorical_feature_names, rank_feature_names и numerical_feature_names).'
            )

        if categorical_feature_names is not None:
            if not isinstance(categorical_feature_names, dict):
                raise ValueError(
                    '`categorical_feature_names` должен представлять собой словарь, в котором ключ '
                    'представляет собой название категориального признака, а значение - список '
                    'возможных его значений.'
                )
            for categorical_feature_name, value_list in categorical_feature_names.items():
                if not isinstance(categorical_feature_name, str):
                    raise ValueError(
                        'Ключи в `categorical_feature_names` должны представлять собой строки. '
                        f'`{categorical_feature_names}` - не строка.'
                    )
                if not isinstance(value_list, list):
                    raise ValueError(
                        'Значения в `categorical_feature_names` должны представлять собой списки '
                        f'строк. Значение `{categorical_feature_name}: {value_list}` - не список.'
                    )
                for value in value_list:
                    if not isinstance(value, str):
                        raise ValueError(
                            'Значения в `categorical_feature_names` должны представлять собой '
                            f'списки строк. Значение `{categorical_feature_name}: {value_list}` - '
                            f'не список строк, `{value}` - не строка.'
                        )

        if rank_feature_names is not None:
            if not isinstance(rank_feature_names, dict):
                raise ValueError(
                    '`rank_feature_names` должен представлять собой словарь, в котором ключ '
                    'представляет собой название рангового признака, а значение - упорядоченный '
                    'список его значений.'
                )
            for rank_feature_name, value_list in rank_feature_names.items():
                if not isinstance(rank_feature_name, str):
                    raise ValueError(
                        'Ключи в `rank_feature_names` должны представлять собой строки. '
                        f'`{rank_feature_name}` - не строка.'
                    )
                if not isinstance(value_list, list):
                    raise ValueError(
                        'Значения в `rank_feature_names` должны представлять собой списки. '
                        f'Значение `{rank_feature_name}: {value_list}` - не список.'
                    )

        if numerical_feature_names is not None:
            if not isinstance(numerical_feature_names, list):
                raise ValueError(
                    '`numerical_feature_names` должен представлять собой список строк.'
                )
            for numerical_feature_name in numerical_feature_names:
                if not isinstance(numerical_feature_name, str):
                    raise ValueError(
                        '`numerical_feature_names` должен представлять собой список строк. '
                        f'`{numerical_feature_name}` - не строка.'
                    )

        if special_cases is not None:
            if not isinstance(special_cases, dict):
                raise ValueError(
                    '`special_cases` должен представлять собой словарь, в котором ключи - строки, '
                    'а значения - либо строки, либо списки строк.'
                )
            for key, value in special_cases.items():
                if not isinstance(key, str):
                    raise ValueError(
                        'special_cases должен представлять собой словарь, в котором ключи - '
                        'строки, а значения - либо строки, либо списки строк.'
                    )
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

        setted_feature_names = []
        if categorical_feature_names:
            for feature_name in categorical_feature_names:
                if feature_name not in X.columns:
                    raise ValueError(
                        f'`categorical_feature_names` содержит признак {feature_name}, которого '
                        'нет в обучающих данных.'
                    )
            setted_feature_names += categorical_feature_names
        if rank_feature_names:
            for feature_name in rank_feature_names.keys():
                if feature_name not in X.columns:
                    raise ValueError(
                        f'`rank_feature_names` содержит признак {feature_name}, которого нет в '
                        'обучающих данных.'
                    )
            setted_feature_names += list(rank_feature_names.keys())
        if numerical_feature_names:
            for feature_name in numerical_feature_names:
                if feature_name not in X.columns:
                    raise ValueError(
                        f'`numerical_feature_names` содержит признак {feature_name}, которого нет '
                        'в обучающих данных.'
                    )
            setted_feature_names += numerical_feature_names
        for feature_name in X.columns:
            if feature_name not in setted_feature_names:
                raise ValueError(
                    f'Обучающие данные содержат признак `{feature_name}`, который не определён ни '
                    'в `categorical_feature_names`, ни в `rank_feature_names`, ни в '
                    '`numerical_feature_names`.'
                )

        self.__feature_names = list(X.columns)
        self.__class_names = sorted(y.unique())
        if categorical_feature_names:
            self.__cat_feature_names = categorical_feature_names
        if rank_feature_names:
            self.__rank_feature_names = rank_feature_names
        if numerical_feature_names:
            self.__num_feature_names = numerical_feature_names

        self.__total_samples_num = X.shape[0]
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
                else:
                    assert False, 'пришли туда, куда были не должны прийти'

        self.__tree = self.__generate_node(
            X, y,
            feature_value=None,
            depth=1,
            available_feature_names=available_feature_names,
            special_cases=special_cases,
        )

    @counter
    def __generate_node(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_value: None | List[str],
            depth: int,
            available_feature_names: List[str],
            special_cases: Optional[Dict[str, str | Dict]] = None,
    ) -> Node:
        """
        Рекурсивная функция создания узлов дерева.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            feature_value: значение признака, по которому этот узел был сформирован.
            depth: глубина узла.
            available_feature_names: список доступных признаков для разбиения входного множества.
            special_cases: словарь {признак, который должен быть первым: признак, который может быть
              после}.

        Returns:
            node: узел дерева.
        """
        impurity = self.__impurity(y)
        samples_num = X.shape[0]
        distribution = self.__distribution(y)
        label = y.value_counts().index[0]

        childs = []
        feature = None
        if samples_num >= self.__min_samples_split and \
                (not self.__max_depth or (depth <= self.__max_depth)):
            feature, xs, ys, feature_values, inf_gain = self.__split(X, y, available_feature_names)

            if feature:
                available_feature_names = available_feature_names.copy()

                self.__feature_importances[feature] += \
                    (samples_num / self.__total_samples_num) * inf_gain

                # добавление открывшихся признаков
                if special_cases:
                    special_cases = special_cases.copy()
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

        node = Node(feature, feature_value, impurity, samples_num, distribution, label, childs)

        return node

    def __distribution(self, y: pd.Series) -> List[int]:
        """Подсчитывает распределение точек данных по классам."""
        distribution = [(y == class_name).sum() for class_name in self.__class_names]

        return distribution

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
    ) -> Tuple[str, List[pd.DataFrame], List[pd.Series], Tuple, float]:
        """
        Разделяет входное множество наилучшим образом.

        Args:
            X: pd.DataFrame с точками данных.
            Y: pd.Series с соответствующими метками.
            available_feature_names: список доступных признаков для разбиения входного множества.

        Returns:
            Кортеж `(feature_name, xs, ys, feature_values, inf_gain)`.
              feature_name: признак, по которому лучше всего разбивать входное множество.
              xs: список DataFrame'ов с точками данных дочерних подмножеств.
              ys: список Series с соответствующими метками дочерних подмножеств.
              feature_values: значения признаков, соответствующие дочерним подмножествам.
              inf_gain: прирост информативности после разбиения.
        """
        best_feature_name = None
        best_xs = []
        best_ys = []
        best_feature_values = tuple()
        best_inf_gain = 0
        for feature_name in available_feature_names:
            if feature_name in self.__cat_feature_names:
                inf_gain, xs, ys, feature_values = self.__best_categorical_split(X, Y, feature_name)
            elif feature_name in self.__rank_feature_names:
                inf_gain, xs, ys, feature_values = self.__best_rank_split(X, Y, feature_name)
            elif feature_name in self.__num_feature_names:
                inf_gain, xs, ys, feature_values = self.__numerical_split(X, Y, feature_name)
            else:
                assert False, 'пришли сюда'

            if inf_gain >= self.__min_impurity_decrease and inf_gain > best_inf_gain:
                best_inf_gain = inf_gain
                best_feature_name = feature_name
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        for list_ in best_feature_values:
            assert isinstance(list_, list)

        return best_feature_name, best_xs, best_ys, best_feature_values, best_inf_gain

    def __best_categorical_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_name: str,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series], Tuple]:
        """
        Разделяет входное множество по категориальному признаку наилучшим образом.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            Кортеж `(inf_gain, xs, ys, feature_values)`.
              inf_gain: прирост информативности при разделении.
              xs: список DataFrame'ов с точками данных дочерних подмножеств.
              ys: список Series с соответствующими метками дочерних подмножеств.
              feature_values: значения признаков, соответствующие дочерним подмножествам.
        """
        best_inf_gain = 0
        best_xs = []
        best_ys = []
        best_feature_values = tuple()

        available_feature_values = set(X[feature_name].tolist())
        if np.NaN in available_feature_values:
            available_feature_values.remove(np.NaN)
        if len(available_feature_values) <= 1:
            return best_inf_gain, best_xs, best_ys, best_feature_values
        available_feature_values = sorted(list(available_feature_values))

        assert len(available_feature_values) != 0, 'добрый почантек'

        # получаем список всех возможных разбиений
        partitions = [tuple(i) for i in categorical_partition(available_feature_values)]
        partitions = partitions[1:]  # убираем вариант без разбиения
        partitions = sorted(partitions, key=len)
        if self.__max_childs:
            partitions = list(filter(lambda x: len(x) <= self.__max_childs, partitions))

        for feature_values in partitions:
            inf_gain, xs, ys = self.__categorical_split(X, y, feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        return best_inf_gain, best_xs, best_ys, best_feature_values

    def __categorical_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_name: str,
            feature_values: Tuple,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series]]:
        """
        Разделяет входное множество по категориальному признаку согласно заданным значениям.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            feature_name: признак, по которому нужно разделить входное множество.
            feature_values: значения признаков, соответствующие дочерним подмножествам.

        Returns:
            Кортеж `(inf_gain, xs, ys)`.
              inf_gain: прирост информативности при разделении.
              xs: список DataFrame'ов с точками данных дочерних подмножеств.
              ys: список Series с соответствующими метками дочерних подмножеств.
        """
        nan_x = X[X[feature_name].isna()]
        nan_y = y[X[feature_name].isna()]

        xs = []
        ys = []
        for list_ in feature_values:
            mask = X[feature_name].isin(list_)
            x = pd.concat([X[mask], nan_x])
            y = pd.concat([y[mask], nan_y])
            if y.shape[0] < self.__min_samples_leaf:
                return 0, [], []
            else:
                xs.append(x)
                ys.append(y)

        inf_gain = self.__information_gain(y, ys)

        return inf_gain, xs, ys

    def __best_rank_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_name: str,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series], Tuple]:
        """Разделяет входное множество по ранговому признаку наилучшим образом."""
        available_feature_values = self.__rank_feature_names[feature_name]

        best_inf_gain = 0
        best_xs = []
        best_ys = []
        best_feature_values = tuple()
        for feature_values in rank_partition(available_feature_values):
            inf_gain, xs, ys = self.__rank_split(X, y, feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        return best_inf_gain, best_xs, best_ys, best_feature_values

    def __rank_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_name: str,
            feature_values: Tuple[List[str], List[str]],
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series]]:
        """Разделяет входное множество по ранговому признаку согласно заданным значениям."""
        nan_x = X[X[feature_name].isna()]
        nan_y = y[X[feature_name].isna()]

        left_list_, right_list_ = feature_values

        left_mask = X[feature_name].isin(left_list_)
        right_mask = X[feature_name].isin(right_list_)

        left_x = pd.concat([X[left_mask], nan_x])
        left_y = pd.concat([y[left_mask], nan_y])
        right_x = pd.concat([X[right_mask], nan_x])
        right_y = pd.concat([y[right_mask], nan_y])

        if left_y.shape[0] < self.__min_samples_leaf or right_y.shape[0] < self.__min_samples_leaf:
            return 0, [], []
        else:
            xs = [left_x, right_x]
            ys = [left_y, right_y]

        inf_gain = self.__information_gain(y, ys)

        return inf_gain, xs, ys

    def __numerical_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_name: str,
    ) -> Tuple[float, List[pd.DataFrame], List[pd.Series], Tuple]:
        """
        Разделяет входное множество по численному признаку, выбирая наилучший порог.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            Кортеж `(inf_gain, xs, ys, feature_values)`.
              inf_gain: прирост информативности при разделении.
              xs: список DataFrame'ов с точками данных дочерних подмножеств.
              ys: список Series с соответствующими метками дочерних подмножеств.
              feature_values: значения признаков, соответствующие дочерним подмножествам.
        """
        nan_x = X[X[feature_name].isna()]
        nan_y = y[X[feature_name].isna()]

        points = sorted(X.loc[X[feature_name].notna(), feature_name].tolist())
        thresholds = [(points[i] + points[i + 1]) / 2 for i in range(len(points) - 1)]
        best_inf_gain = 0
        best_xs = []
        best_ys = []
        best_feature_values = tuple()
        for threshold in thresholds:
            x_less = pd.concat([X[X[feature_name] <= threshold], nan_x])
            y_less = pd.concat([y[X[feature_name] <= threshold], nan_y])

            x_more = pd.concat([X[X[feature_name] > threshold], nan_x])
            y_more = pd.concat([y[X[feature_name] > threshold], nan_y])

            xs = [x_less, x_more]
            ys = [y_less, y_more]
            feature_values = [f'<= {threshold}'], [f'> {threshold}']

            if y_less.shape[0] < self.__min_samples_leaf or \
                    y_more.shape[0] < self.__min_samples_leaf:
                continue

            inf_gain = self.__information_gain(y, ys)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_xs = xs
                best_ys = ys
                best_feature_values = feature_values

        return best_inf_gain, best_xs, best_ys, best_feature_values

    def __information_gain(
            self,
            y: pd.Series,
            ys: List[pd.Series],
    ) -> float:
        """
        Считает прирост информативности.

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
            y: Series с метками родительского множества.
            ys: список Series с метками дочерних подмножеств.

        Returns:
            inf_gain: прирост информативности.
        """
        A = y.shape[0]

        second_term = 0
        for y_i in ys:
            A_i = y_i.shape[0]
            second_term += (A_i/A) * self.__impurity(y_i)

        inf_gain = self.__impurity(y) - second_term

        assert isinstance(inf_gain, float), f'бам-бам, бим-бим {type(inf_gain)}'

        return inf_gain

    def get_params(
            self,
            deep: Optional[bool] = True,  # реализован для sklearn.model_selection.GridSearchCV
    ) -> Dict:
        """Возвращает параметры этого классификатора."""
        params = {
            'max_depth': self.__max_depth,
            'criterion': self.__criterion,
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
                    f'Недопустимый параметр {param} для дерева {self}. Проверьте список доступных '
                    'параметров с помощью `estimator.get_params().keys()`.'
                )

            setattr(self, param, value)
            valid_params[param] = value

        return self

    def predict(self, X: pd.DataFrame | pd.Series) -> List[str] | str:
        """Предсказывает метки классов для точек данных в X."""
        if isinstance(X, pd.DataFrame):
            y_pred = [self.predict(point) for _, point in X.iterrows()]
        elif isinstance(X, pd.Series):
            y_pred = self.__predict(self.__tree, X)
        else:
            raise ValueError('X должен представлять собой pd.DataFrame или pd.Series.')

        assert y_pred is not None, 'предсказывает None'

        return y_pred

    def __predict(self, node: Node, point: pd.Series) -> str:
        """Предсказывает метку класса для точки данных."""
        Y = None
        # если мы дошли до листа
        if node.feature is None:
            Y = node.label
            assert Y is not None, 'label оказался None'
        elif node.feature in self.__cat_feature_names | self.__rank_feature_names:
            # ищем ту ветвь, по которой нужно идти
            for child in node.childs:
                if child.feature_value == point[node.feature]:
                    Y = self.__predict(child, point)
                    break
            else:
                # если такой ветви нет
                if Y is None:
                    Y = node.label
        elif node.feature in self.__num_feature_names:
            # ищем ту ветвь, по которой нужно идти
            threshold = float(node.childs[0].feature_value[0][3:])
            if point[node.feature] <= threshold:
                Y = self.__predict(node.childs[0], point)
            elif point[node.feature] > threshold:
                Y = self.__predict(node.childs[1], point)
            else:
                assert False, 'пришли сюда'
        else:
            assert False, ('node.split_feature и не None, и не в `categorical_feature_names` и не в'
                           '`numerical_feature_names`')

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
        """
        Визуализирует дерево решений.

        Если указаны именованные параметры, сохраняет визуализацию в виде файла(ов).

        Args:
            rounded: скруглять ли углы у узлов (они в форме прямоугольника).
            show_impurity: показывать ли загрязнённость узла.
            show_num_samples: показывать ли количество точек в узле.
            show_distribution: показывать ли распределение точек по классам.
            show_label: показывать ли класс, к которому относится узел.
            **kwargs: аргументы для graphviz.Digraph.render.

        Returns:
            Объект класса Digraph, содержащий описание графовой структуры дерева для визуализации.
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

        node_content = []
        if node.feature:
            node_content.append(f'{node.feature}')
        if show_impurity:
            node_content.append(f'{self.__criterion} = {node.impurity:.3f}')
        if show_num_samples:
            node_content.append(f'samples = {node.samples}')
        if show_distribution:
            node_content.append(f'distribution: {node.distribution}')
        if show_label:
            node_content.append(f'label = {node.label}')
        node_content = '\n'.join(node_content)

        self.__graph.node(name=node_name, label=node_content)
        if parent_name:
            if isinstance(node.feature_value, list):
                a = [str(i) for i in node.feature_value]
                node_label = '\n'.join(a)
            else:
                assert False, 'пришли сюда'
            self.__graph.edge(parent_name, node_name, label=node_label)

        for child in node.childs:
            self.__add_node(
                child, node_name, show_impurity, show_num_samples, show_distribution, show_label
            )

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Возвращает точность по заданным тестовым данным и меткам."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X должен представлять собой pd.DataFrame.')

        if not isinstance(y, pd.Series):
            raise ValueError('y должен представлять собой pd.Series.')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X и y должны быть одной длины.')

        score = accuracy_score(y, self.predict(X))

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
