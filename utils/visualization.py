import numpy as np
import matplotlib.pyplot as plt


def get_confusion_matrix(ys_true, ys_pred, true_labels, pred_labels):
    """Подсчитывает матрицу ошибок.

    :param ys_true: метки абсолютной истины.
    :param ys_pred: предсказанные метки.
    :param true_labels: множество меток абсолютной истины.
    :param pred_labels: множество предсказываемых меток.
    :return: матрица ошибок.
    """
    confusion_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype='int64')
    for y_true, y_pred in zip(ys_true, ys_pred):
        confusion_matrix[true_labels.index(y_true)][pred_labels.index(y_pred)] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, true_labels, pred_labels,
                          threshold=None):
    """Визуализирует матрицу ошибок.

    :param confusion_matrix: матрица ошибок.
    :param true_labels: множество меток абсолютной истины.
    :param pred_labels: множество предсказываемых меток.
    :param threshold: порог принятия решения.
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
