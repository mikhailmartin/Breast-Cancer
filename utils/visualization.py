"""Здесь содержатся разнообразные утилиты для визуализации."""
from typing import Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
import seaborn as sns
sns.set_theme()

import utils


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
    if df[feature_name].isna().any():
        df[feature_name].replace({np.NaN: '-'}, inplace=True)
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
        xlim: Tuple[int | float, int | float],
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


def num_feature_report(
        data: pd.DataFrame,
        *,
        feature_colname: str,
        target_colname: str,
        value_range: Optional[Tuple[float | None, float | None]] = (None, None),
        figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
        histplot_args: Dict = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Визуализирует разницу в распределениях численного непрерывного признака для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        value_range: задаваемый диапазон рассматриваемых значений.
        figsize: (ширина, высота) рисунка в дюймах.
        histplot_args: аргументы для seaborn.histplot().

    Returns:
        Кортеж `(fig, axes)`.
          fig: matplotlib.Figure, содержащий все графики.
          axes: matplotlib.axes.Axes, содержащие отрисованные график.
    """
    # подготовка данных
    data = data[[feature_colname, target_colname]].copy()
    data = slice_by_value_range(data, feature_colname, value_range)

    if histplot_args is None:
        histplot_args = {}

    if data[feature_colname].isna().sum():
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        na_bar_plot(data, feature_colname=feature_colname, target_colname=target_colname, ax=axes[2])
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Violinplot
    sns.violinplot(
        data=data,
        x=feature_colname,
        y=target_colname,
        orient='h',
        ax=axes[0],
    )
    axes[0].set(title='Violinplot')

    # Density Histogram
    sns.histplot(
        data=data,
        x=feature_colname,
        hue=target_colname,
        stat='density',
        common_norm=False,
        ax=axes[1],
        **histplot_args,
    )
    axes[1].set(title='Density Histogram')

    fig.suptitle(f'num_feature_report для признака {feature_colname}')

    labels = data[target_colname].unique()
    results = stats.ttest_ind(
        data.loc[
            (data[target_colname] == labels[0]) & data[feature_colname].notna(), feature_colname],
        data.loc[
            (data[target_colname] == labels[1]) & data[feature_colname].notna(), feature_colname],
    )

    if results.pvalue < .05:
        print(f't-критерий Стьюдента: Выборки различимы (p-значение: {results.pvalue:.3f})')
    else:
        print(f't-критерий Стьюдента: Выборки неразличимы (p-значение: {results.pvalue:.3f})')

    return fig, axes


def cat_feature_report(
        data: pd.DataFrame,
        *,
        feature_colname: str,
        target_colname: str,
        figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]:
    """
    Визуализирует разницу в распределениях категориального признака для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        figsize: (ширина, высота) рисунка в дюймах.

    Returns:
        Кортеж `(fig, ax)`.
          fig: matplotlib.Figure, содержащий все графики.
          axes: matplotlib.axes.Axes, содержащие отрисованные график.
    """
    data = data[[feature_colname, target_colname]].copy()

    if data[feature_colname].isna().sum():
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        na_bar_plot(
            data, feature_colname=feature_colname, target_colname=target_colname, ax=axes[1])
        data = data[data[feature_colname].notna()]
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]  # заглушка

    bar_plot(data, feature_colname=feature_colname, target_colname=target_colname, ax=axes[0])
    axes[0].set(
        xlabel='Целевые классы',
        ylabel='Доли категорий',
    )
    fig.suptitle(f'cat_feature_report для признака {feature_colname}')

    return fig, axes


def na_bar_plot(
        data: pd.DataFrame,
        *,
        feature_colname: str,
        target_colname: str,
        ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Визуализирует разницу в доле пропусков для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        ax: matplotlib.axes.Axes, на котором следует отрисовать график.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    data = data[[feature_colname, target_colname]].copy()

    if not ax:
        ax = plt.subplot()

    data['has_na'] = data[feature_colname].isna().replace({True: 'пропуск', False: 'значение'})
    del data[feature_colname]

    bar_plot(data, feature_colname='has_na', target_colname=target_colname, ax=ax)
    ax.set(ylabel='Доли пропусков')

    return ax


def bar_plot(
        data: pd.DataFrame,
        *,
        feature_colname: str,
        target_colname: str,
        ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Визуализирует распределение значений признака в виде столбчатой диаграммы для целевых классов.

    Args:
        data: pd.DataFrame, содержащий исследуемый признак и целевую переменную.
        feature_colname: название столбца с исследуемым признаком.
        target_colname: название столбца с целевой переменной.
        ax: ранее созданный ax.

    Returns:
        ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    data = data[[feature_colname, target_colname]].copy()

    labels = sorted(data[target_colname].unique())
    categories = sorted(data[feature_colname].unique())

    if not ax:
        ax = plt.subplot()

    a = []
    for category in categories:
        category_ratios = []
        for label in labels:
            class_df = data[data[target_colname] == label]
            all_count = class_df.shape[0]
            category_count = class_df[class_df[feature_colname] == category].shape[0]
            ratio = category_count / all_count
            category_ratios.append(ratio)

        all_count = data.shape[0]
        category_count = data[data[feature_colname] == category].shape[0]
        ratio = category_count / all_count
        category_ratios.append(ratio)

        a.append(category_ratios)

    labels = [str(label) for label in labels] + ['весь датасет']

    bottom = np.zeros(len(labels))
    for i, category in enumerate(categories):
        ax.bar(labels, a[i], bottom=bottom, label=category)
        bottom += a[i]

    # add annotations
    for c in ax.containers:

        # customize the label to account for cases when there might not be a bar section
        labels = [f'{h:2.3%}' if (h := v.get_height()) > .05 else '' for v in c]

        # set the bar label
        ax.bar_label(c, labels=labels, label_type='center', fontsize=8)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(bbox_to_anchor=(1, 1.05), title='Категории')
    ax.set(
        xlabel='Целевые классы',
        ylabel='Доли категорий',
    )

    return ax


def slice_by_value_range(
        data: pd.DataFrame,
        feature_colname: str,
        value_range: Optional[Tuple[float | None, float | None]] = (None, None),
):
    """
    Возвращает срез pd.DataFrame по задаваемому признаку и диапазону значений.

    Args:
        data: pd.DataFrame, который необходимо обрезать.
        feature_colname: название столбца с признаком, по которому необходимо сделать срез.
        value_range: задаваемый диапазон рассматриваемых значений.

    Returns:
        data: срезанный pd.DataFrame.
    """
    data = data.copy()

    min_value = value_range[0]
    max_value = value_range[1]
    if min_value:
        data = data[data[feature_colname] >= min_value]
    if max_value:
        data = data[data[feature_colname] <= max_value]

    return data


def plot_distribution(
        num1: int,
        num2: int,
        num3: int
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Распределение точек данных по классам', fontsize=16)
    ax.bar(utils.constants.LABELS, [num1, num2, num3], tick_label=utils.constants.LABELS)
    autolabel(ax, height_factor=0.85)

    plt.show()


def get_accuracy_matrix(
        df: pd.DataFrame,
        question: str
) -> Tuple[List[List[float]], List[str]]:
    """
    Подсчитывает матрицу точностей совпадений ответа на вопрос и диагноза.

    Args:
        df: DataFrame, из которого будет вытаскиваться информация.
        question: вопрос, для которого будет построена матрица точностей.

    Returns:
        Кортеж `(accuracy_matrix, answers)`.
          matrix: собственно матрица точностей.
          answers: ответы на вопрос.
    """
    # временный DataFrame только с ответами на вопрос и метками
    tmp = df[[question, utils.constants.LABEL]]
    # сбрасывание всех NaN
    tmp.dropna()

    accuracy_matrix = []
    answer_set = tmp.value_counts(subset=question).index
    for answer in answer_set:
        row = []
        for label in utils.constants.LABELS:
            tp = tmp[(tmp[question] == answer) & (tmp[utils.constants.LABEL] == label)].shape[0]
            tn = tmp[(tmp[question] != answer) & (tmp[utils.constants.LABEL] != label)].shape[0]
            fp = tmp[(tmp[question] == answer) & (tmp[utils.constants.LABEL] != label)].shape[0]
            fn = tmp[(tmp[question] != answer) & (tmp[utils.constants.LABEL] == label)].shape[0]

            accuracy = (tp + tn) / (tp + tn + fp + fn)

            row.append(accuracy)

        accuracy_matrix.append(row)

    answers = list(answer_set)

    return accuracy_matrix, answers


def plot_accuracy_matrix(
        df: pd.DataFrame,
        question: str
) -> None:
    """
    Визуализирует матрицу точностей.

    Args:
        df: DataFrame, из которого будет вытаскиваться информация.
        question: вопрос, для которого строится матрица.
    """
    matrix, col_labels = get_accuracy_matrix(df, question)

    ax = plt.subplot()

    # визуализируем матрицу в виде тепловой карты
    ax.imshow(matrix)

    ax.set_xticks(range(len(utils.constants.LABELS)))
    ax.set_xticklabels(utils.constants.LABELS)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right', rotation_mode='anchor')

    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(col_labels)

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(len(utils.constants.LABELS) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(matrix) + 1) - .5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_title(question)

    matrix = np.array(matrix)

    def redact(number):
        string = f'{number:.1%}'
        return string

    for i in range(len(matrix)):
        for j in range(len(utils.constants.LABELS)):
            if matrix[i][j] < ((matrix.max() + matrix.min()) / 2):
                ax.text(j, i, redact(matrix[i][j]), ha='center', va='center', color='white')
            else:
                ax.text(j, i, redact(matrix[i][j]), ha='center', va='center', color='black')

    plt.show()


def plot_history(history: tf.keras.callbacks.History, *, dpi: Optional[int] = 70) -> None:
    """
    Визуализирует историю обучения.

    Args:
        history: собственно история.
        dpi: количество пикселей на дюйм.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
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
