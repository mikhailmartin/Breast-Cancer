"""Здесь содержаться утилиты для работы с нейросетями."""
from typing import Dict, List, Tuple

import tensorflow as tf


# здесь названия входов такие тупые, потому что они на русском неправильно читаются
INPUT_NAMES = {
    'input_2': 'numerical',
    'input_3': 'categorical',
    'input_4': 'numerical',
    'input_5': 'categorical',
    'input_6': 'categorical',
    'input_7': 'categorical',
    'input_8': 'categorical',
    'input_9': 'categorical',
    'input_11': 'categorical',
    'input_12': 'categorical',
    'input_13': 'categorical',
    'input_14': 'categorical',
    'input_15': 'categorical',
    'input_16': 'categorical',
    'input_17': 'categorical',
    'input_18': 'categorical',
    'input_19': 'categorical',
    'input_20': 'categorical',
    'input_22': 'numerical',
    'input_23': 'categorical',
    'input_24': 'numerical',
    'input_25': 'categorical',
    'input_26': 'categorical',
    'input_27': 'categorical',
    'input_28': 'categorical',
    'input_29': 'categorical',
    'input_30': 'categorical',
    'input_31': 'categorical',
    'input_32': 'categorical',
    'input_33': 'categorical',
    'input_34': 'categorical',
    'input_35': 'categorical',
    'label': None,
}


def dataset_from_csv(file_path: str) -> tf.data.Dataset:
    """Читает Dataset из CSV-файла. One-hot-encode'ит метки.

    Args:
        file_path: путь до CSV-файла.

    Returns:
        датасет.
    """
    # создаём Dataset, читая CSV-файл
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_path,
        batch_size=1,
        column_names=INPUT_NAMES.keys(),
        label_name='label',
        header=False,
        num_epochs=1,
    )
    # one-hot-encoding метки
    dataset = dataset.map(lambda point, label: (point, tf.one_hot(label, depth=3)))
    dataset.batch(1)

    return dataset


# архитектура нейросети
def create_model(layer_width: int, entire_ds: tf.data.Dataset) -> tf.keras.Model:
    """Создаёт модель.

    Args:
        layer_width: ширина слоёв.
        entire_ds: целый датасет, из которого изучается статистика данных.
    """
    input_names = INPUT_NAMES.copy()
    input_names.pop('label')

    categorical_inputs, numerical_inputs = create_inputs(input_names)
    encoded_categorical_inputs = encode_categorical_inputs(categorical_inputs, entire_ds)
    x = tf.keras.layers.concatenate(encoded_categorical_inputs + numerical_inputs)
    x = tf.keras.layers.Dense(layer_width, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(layer_width, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(categorical_inputs + numerical_inputs, outputs)

    return model


def create_inputs(input_names: Dict[str, str]) -> Tuple[List[tf.keras.Input], List[tf.keras.Input]]:
    """Возвращает 2 списка: из категориальных и численных входов.

    Args:
        input_names: словарь {название_входа: его тип (численный, категориальный)}.

    Returns:
        Кортеж `(categorical_inputs, numerical_inputs)`.
          categorical_inputs: список категориальных входов.
          numerical_inputs: список численных входов.
    """
    categorical_inputs = []
    numerical_inputs = []
    for input_name, input_type in input_names.items():
        if input_type == 'categorical':
            categorical_inputs.append(tf.keras.Input(shape=(1,), name=input_name, dtype='int64'))
        elif input_type == 'numerical':
            numerical_inputs.append(tf.keras.Input(shape=(1,), name=input_name, dtype='float64'))

    return categorical_inputs, numerical_inputs


def encode_categorical_inputs(
        inputs: List[tf.keras.Input],
        entire_ds: tf.data.Dataset,
) -> List[tf.Tensor]:
    """Для категориальных признаков создаёт слой one-hot-encoding.

    Args:
        inputs: список из входов нейросети.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        Список предобработанных признаков.
    """
    encoded_categorical_features = [
        encode_categorical_input(input_, entire_ds)
        for input_ in inputs
        if INPUT_NAMES[input_.name] == 'categorical'
    ]

    return encoded_categorical_features


def encode_categorical_input(input_: tf.keras.Input, entire_ds: tf.data.Dataset) -> tf.Tensor:
    """One-hot-encode'ит категориальный признак.

    Args:
        input_: вход нейросети категориального признака.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        Предобработанный категориальный признак.
    """
    lookup = tf.keras.layers.IntegerLookup(
        output_mode='binary',
        num_oov_indices=0,
        mask_token=-1,
    )
    # Подготавливаем Dataset, который содержит только необходимый признак
    feature_ds = entire_ds.map(lambda x, y: x[input_.name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    # Изучаем набор возможных строковых значений и присваиваем им фиксированный целочисленный индекс
    lookup.adapt(feature_ds)
    # Превращаем строковый вход в целочисленные индексы
    encoded_feature = lookup(input_)

    return encoded_feature
