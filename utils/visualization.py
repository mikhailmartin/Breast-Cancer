"""Здесь содержатся разнообразные утилиты для визуализации."""
from typing import List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import definitions as defs


def autolabel(
        ax: matplotlib.axes.Axes,
        labels: Optional[List[str]] = None,
        height_factor: Optional[float] = 1.01,
) -> None:
    """Подписывает значение столбцов в гистограмме."""
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        if labels is not None:
            try:
                label = labels[i]
            except (TypeError, KeyError):
                label = ' '
        else:
            label = f'{int(height)}'
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height_factor * height,
            f'{label}',
            ha='center',
            va='bottom',
        )


def plot_pies(
        df: pd.DataFrame,
        feature_name: str,
        label_column: str,
        *,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
) -> None:
    """Визуализирует распределение значений признака в виде круговой диаграммы.

    Args:
        df: DataFrame, из которого будет вытаскиваться распределение.
        feature_name: признак, для которого визуализируется распределение ответов.
        label_column: название столбца с метками.
        nrows: количество строчек на рисунке.
        ncols: количество столбцов на рисунке.
    """
    df = df.copy()
    if df[feature_name].isnull().any():
        df.replace({feature_name: {np.NaN: '-'}}, inplace=True)
        values = set(df[feature_name].tolist())
        values.remove('-')
        values = sorted(list(values))
        values.insert(0, '-')
    else:
        values = sorted(list(set(df[feature_name].tolist())))
    labels = sorted(list(set(df[label_column].tolist())))
    if nrows is None or ncols is None:
        nrows = 1
        ncols = len(labels) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5))
    axes = axes.flat
    fig.suptitle(feature_name, fontsize=16)
    # отдельные пироги по классам
    for label, ax in zip(labels, axes[:len(labels)]):
        ax.set_title(label)
        sizes = [
            df[(df[label_column] == label) & (df[feature_name] == value)].shape[0]
            for value in values
        ]
        ax.pie(sizes, autopct='%1.1f%%')
    # пирог для всего датасета
    axes[-1].set_title('целый датасет')
    sizes = [df[df[feature_name] == value].shape[0] for value in values]
    wedges, _, _ = axes[-1].pie(sizes, autopct='%1.1f%%')
    axes[-1].legend(wedges, values, bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()


def plot_hists(
        df: pd.DataFrame,
        feature_name: str,
        label_column: str,
        *,
        bins: Optional[int] = None,
        xlim: Tuple[Union[int, float], Union[int, float]],
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
) -> None:
    """Визуализирует гистограммы значений признака.

    Args:
        df: DataFrame, из которого будет вытаскиваться гистограммы.
        feature_name: вопрос, для которого визуализируются гистограммы.
        label_column: название столбца с метками.
        bins: количество бинов гистограмм.
        xlim: пределы шкалы по x.
        nrows: количество строчек на рисунке.
        ncols: количество столбцов на рисунке.
    """
    labels = sorted(list(set(df[label_column].tolist())))
    if nrows is None or ncols is None:
        nrows = 1
        ncols = len(labels) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5), sharey=True)
    axes = axes.flat
    fig.suptitle(feature_name, fontsize=16)
    # отдельные гистограммы по классам
    for label, ax in zip(labels, axes[:len(labels)]):
        ax.hist(df.loc[df[label_column] == label, feature_name].tolist(), bins=bins)
        ax.set_title(label)
        ax.set_xlim(*xlim)
    # гистограмма для всего датасета
    axes[-1].hist(df[feature_name].tolist(), bins=bins)
    axes[-1].set_title('целый датасет')
    axes[-1].set_xlim(*xlim)

    plt.show()


def plot_distribution(
        num1: int,
        num2: int,
        num3: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Распределение точек данных по классам', fontsize=16)
    ax.bar(defs.LABELS, [num1, num2, num3], tick_label=defs.LABELS)
    autolabel(ax, height_factor=0.85)

    plt.show()


def get_accuracy_matrix(
        df: pd.DataFrame,
        question: str
) -> Tuple[List[List[float]], List[str]]:
    """Подсчитывает матрицу точностей совпадений ответа на вопрос и диагноза.

    Args:
        df: DataFrame, из которого будет вытаскиваться информация.
        question: вопрос, для которого будет построена матрица точностей.

    Returns:
        Кортеж `(accuracy_matrix, answers)`.
          matrix: собственно матрица точностей.
          answers: ответы на вопрос.
    """
    # временный DataFrame только с ответами на вопрос и метками
    tmp = df[[question, defs.LABEL]]
    # сбрасывание всех NaN
    tmp.dropna()

    accuracy_matrix = []
    answer_set = tmp.value_counts(subset=question).index
    for answer in answer_set:
        row = []
        for label in defs.LABELS:
            tp = tmp[(tmp[question] == answer) & (tmp[defs.LABEL] == label)].shape[0]
            tn = tmp[(tmp[question] != answer) & (tmp[defs.LABEL] != label)].shape[0]
            fp = tmp[(tmp[question] == answer) & (tmp[defs.LABEL] != label)].shape[0]
            fn = tmp[(tmp[question] != answer) & (tmp[defs.LABEL] == label)].shape[0]

            accuracy = (tp + tn) / (tp + tn + fp + fn)

            row.append(accuracy)

        accuracy_matrix.append(row)

    answers = list(answer_set)

    return accuracy_matrix, answers


def plot_accuracy_matrix(
        df: pd.DataFrame,
        question: str
) -> None:
    """Визуализирует матрицу точностей.

    Args:
        df: DataFrame, из которого будет вытаскиваться информация.
        question: вопрос, для которого строится матрица.
    """
    matrix, col_labels = get_accuracy_matrix(df, question)

    ax = plt.subplot()

    # визуализируем матрицу в виде тепловой карты
    ax.imshow(matrix)

    ax.set_xticks(range(len(defs.LABELS)))
    ax.set_xticklabels(defs.LABELS)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')

    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(col_labels)

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(len(defs.LABELS) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(matrix) + 1) - .5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_title(question)

    matrix = np.array(matrix)

    def redact(number):
        string = f'{number:.1%}'
        return string

    for i in range(len(matrix)):
        for j in range(len(defs.LABELS)):
            if matrix[i][j] < ((matrix.max() + matrix.min()) / 2):
                ax.text(j, i, redact(matrix[i][j]), ha='center', va='center', color='white')
            else:
                ax.text(j, i, redact(matrix[i][j]), ha='center', va='center', color='black')

    plt.show()


def get_confusion_matrix(
        ys_true: List[str],
        ys_pred: List[str],
        true_labels: List[str],
        pred_labels: List[str],
) -> np.ndarray:
    """Подсчитывает матрицу ошибок.

    Args:
        ys_true: метки абсолютной истины.
        ys_pred: предсказанные метки.
        true_labels: множество меток абсолютной истины.
        pred_labels: множество предсказываемых меток.

    Returns:
        Матрица ошибок.
    """
    confusion_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype='int64')
    for y_true, y_pred in zip(ys_true, ys_pred):
        confusion_matrix[true_labels.index(y_true)][pred_labels.index(y_pred)] += 1

    return confusion_matrix


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        true_labels: List[str],
        pred_labels: List[str],
        threshold: Optional[float] = None,
) -> None:
    """Визуализирует матрицу ошибок.

    Args:
        confusion_matrix: матрица ошибок.
        true_labels: множество меток абсолютной истины.
        pred_labels: множество предсказываемых меток.
        threshold: порог принятия решения.
    """
    ax = plt.subplot()
    ax.imshow(confusion_matrix)
    # подписываем метки на оси Y
    ax.set_yticks(range(confusion_matrix.shape[0]))
    ax.set_yticklabels(true_labels)
    ax.set_ylabel('Правильные метки')
    # прописываем метки на оси X
    ax.set_xticks(range(confusion_matrix.shape[1]))
    ax.set_xticklabels(pred_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Предсказанные метки')
    # прописываем циферки
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if confusion_matrix[i][j] < ((confusion_matrix.max() + confusion_matrix.min()) / 2):
                ax.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color='yellow')
            else:
                ax.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color='black')
    if threshold:
        ax.set_title(f'Порог принятия решения: {threshold:2.0%}')

    plt.show()


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Визуализирует историю обучения.

    Args:
        history: собственно история.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(history.history['accuracy'], label='train_accuracy')
    axes[0].plot(history.history['val_accuracy'], label='val_accuracy')
    axes[0].set_xlabel('эпохи')
    axes[0].set_ylabel('accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='train_loss')
    axes[1].plot(history.history['val_loss'], label='val_loss')
    axes[1].set_xlabel('эпохи')
    axes[1].set_ylabel('loss')
    axes[1].legend()

    plt.show()


def plot_scatter(dataframe, feature1, feature2, *, target=None):
    """Отрисовывает диаграмму рассеяния для двух признаков."""
    ax = plt.subplot()

    if target:
        labels = sorted(list(set(dataframe[target])))
        colors = [plt.cm.tab10(i / float(len(labels) - 1)) for i in range(len(labels))]
        for label, color in zip(labels, colors):
            mask = (dataframe[target] == label)
            ax.scatter(
                dataframe.loc[mask, feature1],
                dataframe.loc[mask, feature2],
                marker='.',
                color=color,
                label=label,
                )
            ax.legend()
    else:
        ax.scatter(dataframe[feature1], dataframe[feature2], marker='.')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    plt.show()
