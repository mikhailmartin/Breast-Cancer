"""Здесь содержатся разнообразные утилиты для визуализации."""
import matplotlib.pyplot as plt
import numpy as np

from utils import definitions as defs


def autolabel(ax, labels=None, height_factor=1.01):
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
            va='bottom')


def plot_pies(dataframe, question):
    """Визуализирует распределение ответов на вопрос в виде круговой диаграммы.

    Args:
        dataframe: DataFrame, из которого будет вытаскиваться распределение.
        question: вопрос, для которого визуализируется распределение ответов.
    """
    df = dataframe.copy()
    if df[question].isnull().sum():
        df.replace({question: {np.NaN: '-'}}, inplace=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(question, fontsize=16)
    # отдельные пироги по классам
    for label, ax in zip(defs.LABELS, axes[:3]):
        ax.set_title(label)
        values = set(df[question].tolist())
        sizes = [
            df[(df[defs.LABEL] == label) & (df[question] == value)].shape[0]
            for value in values
        ]
        ax.pie(sizes, autopct='%1.1f%%')
    # пирог для всего датасета
    axes[3].set_title('целый датасет')
    values = set(df[question].tolist())
    sizes = [df[df[question] == value].shape[0] for value in values]
    wedges, _, _ = axes[3].pie(sizes, autopct='%1.1f%%')
    axes[3].legend(wedges, values, bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()


def plot_hists(df, question, suptitle, bins, xlim):
    """Визуализирует гистограммы ответа на вопрос.

    Args:
        df: DataFrame, из которого будет вытаскиваться гистограммы.
        question: вопрос, для которого визуализируются гистограммы.
        suptitle: подзаголовок к рисунку.
        bins: количество бинов гистограмм.
        xlim: предел шкалы по x.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    fig.suptitle(suptitle, fontsize=16)
    # отдельные гистограммы по классам
    for label, ax in zip(defs.LABELS, axes[:3]):
        ax.hist(df[df[defs.LABEL] == label][question].tolist(), bins=bins, range=(0, bins))
        ax.set_title(label)
        ax.set_xlim(0, xlim)
    # гистограмма для всего датасета
    axes[3].hist(df[question].tolist(), bins=bins, range=(0, bins))
    axes[3].set_title('целый датасет')
    axes[3].set_xlim(0, xlim)

    plt.show()


def plot_distribution(n1_old, n2_old, n3_old, n1_new, n2_new, n3_new):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    fig.suptitle('Распределение точек данных по классам', fontsize=16)
    axes[0].set_title('до очистки')
    axes[0].bar(defs.LABELS, [n1_old, n2_old, n3_old], tick_label=defs.LABELS)
    autolabel(axes[0], height_factor=0.85)

    axes[1].set_title('после очистки')
    axes[1].bar(defs.LABELS, [n1_new, n2_new, n3_new], tick_label=defs.LABELS)
    autolabel(axes[1], height_factor=0.85)

    plt.show()


def get_accuracy_matrix(df, question):
    """Подсчитывает матрицу точностей совпадений ответа на вопрос и диагноза.

    Args:
        df: DataFrame, из которого будет вытаскиваться информация.
        question: вопрос, для которого будет построена матрица точностей.

    Returns:
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


def plot_accuracy_matrix(df, question):
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


def get_confusion_matrix(ys_true, ys_pred, true_labels, pred_labels):
    """Подсчитывает матрицу ошибок.

    Args:
        ys_true: метки абсолютной истины.
        ys_pred: предсказанные метки.
        true_labels: множество меток абсолютной истины.
        pred_labels: множество предсказываемых меток.

    Returns:
        confusion_matrix: матрица ошибок.
    """
    confusion_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype='int64')
    for y_true, y_pred in zip(ys_true, ys_pred):
        confusion_matrix[true_labels.index(y_true)][pred_labels.index(y_pred)] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, true_labels, pred_labels, threshold=None):
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


def plot_history(history):
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
