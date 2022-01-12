"""Здесь содержаться утилиты для работы с нейросетями."""
import tensorflow as tf

from . import definitions as defs


def dataset_from_csv(file_path):
    """Читает Dataset из CSV-файла. One-hot-encode'ит метки.

    Args:
        file_path: путь до CSV-файла.

    Returns:
        dataset: датасет.
    """
    # создаём Dataset, читая CSV-файл
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_path,
        batch_size=1,
        column_names=defs.INPUT_NAMES.keys(),
        label_name='label',
        header=False,
        num_epochs=1
    )
    # one-hot-encoding метки
    dataset = dataset.map(lambda point, label: (point, tf.one_hot(label, depth=3)))
    dataset.batch(1)

    return dataset
