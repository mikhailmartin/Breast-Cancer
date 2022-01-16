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


def plot_pies(df, question, ignore=None):
    """Визуализирует распределение ответов на вопрос в виде круговой диаграммы.

    Args:
        df: DataFrame, из которого будет вытаскиваться распределение.
        question: вопрос, для которого визуализируется распределение ответов.
        ignore: множество ответов, которые стоит проигнорировать при визуализации.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    for label, ax in zip(defs.LABELS, axes):
        sizes = []
        answer_set = set(df[question].tolist())
        # удаление ненужного куска(ов)
        if ignore is None:
            pass
        elif isinstance(ignore, set):
            for ignore_answer in ignore:
                answer_set.remove(ignore_answer)
        else:
            answer_set.remove(ignore)

        for answer in answer_set:
            amount = df[(df['Метка'] == label) & (df[question] == answer)].shape[0]
            sizes.append(amount)
        wedges, texts, _ = ax.pie(sizes, autopct='%1.1f%%')
        ax.set_title(label)
        if ax is axes[-1]:
            ax.legend(wedges, answer_set, bbox_to_anchor=(1, 0, 0.5, 1))

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
    tmp = df[[question, 'Метка']]
    # сбрасывание всех NaN
    tmp.dropna()

    accuracy_matrix = []
    answer_set = tmp.value_counts(subset=question).index
    for answer in answer_set:
        row = []
        for label in defs.LABELS:
            tp = tmp[(tmp[question] == answer) & (tmp['Метка'] == label)].shape[0]
            tn = tmp[(tmp[question] != answer) & (tmp['Метка'] != label)].shape[0]
            fp = tmp[(tmp[question] == answer) & (tmp['Метка'] != label)].shape[0]
            fn = tmp[(tmp[question] != answer) & (tmp['Метка'] == label)].shape[0]

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


def plot_history(history, mode):
    """Визуализирует историю обучения.

    Args:
        history: собственно история.
        mode: {'accuracy', 'loss'} что отображать точность или потери.
    """
    ax = plt.subplot()
    ax.plot(history.history[f'{mode}'])
    ax.plot(history.history[f'val_{mode}'])
    ax.set_xlabel('epochs')
    ax.set_ylabel(f'{mode}')
    ax.legend(['train', 'val'])

    plt.show()
