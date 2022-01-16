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
        num_epochs=1)
    # one-hot-encoding метки
    dataset = dataset.map(lambda point, label: (point, tf.one_hot(label, depth=3)))
    dataset.batch(1)

    return dataset


# архитектура нейросети
def create_inputs(input_names):
    """Создаёт список из входов.

    Args:
        input_names: словарь {название_входа: его тип (численный, категориальный)}.

    Returns:
        inputs: список входов
    """
    inputs = [
        tf.keras.Input(shape=(1,), name=input_name, dtype='float64')
        if input_names[input_name] == 'numerical'
        else tf.keras.Input(shape=(1,), name=input_name, dtype='int64')
        for input_name in input_names
    ]

    return inputs


def encode_numerical_input(inpt, entire_ds):
    """Реализует предобработку численного признака.

    Предобработка заключается в создании слоя нормализации, который вычитает среднее арифметическое
    и масштабирует признак до диапазона [-1, 1].

    Args:
        inpt (tf.keras.Input): вход нейросети численного признака.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        encoded_feature: предобработанный численный признак.
    """
    # Создание слоя нормализации для признака
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    # Подготавливаем Dataset, который содержит только необходимый признак
    feature_ds = entire_ds.map(lambda x, y: x[inpt.name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    # Изучается статистика данных
    normalizer.adapt(feature_ds)
    # Нормализация входного признака
    encoded_feature = normalizer(inpt)

    return encoded_feature


def encode_categorical_input(inpt, entire_ds):
    """One-hot-encode'ит категориальный признак.

    Args:
        inpt (tf.keras.Input): вход нейросети численного признака.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        encoded_feature: предобработанный категориальный признак.
    """
    lookup = tf.keras.layers.experimental.preprocessing.IntegerLookup(
        output_mode='binary',
        num_oov_indices=0,
        mask_token=-1)
    # Подготавливаем Dataset, который содержит только необходимый признак
    feature_ds = entire_ds.map(lambda x, y: x[inpt.name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    # Изучаем набор возможных строковых значений и присваиваем им фиксированный целочисленный индекс
    lookup.adapt(feature_ds)
    # Превращаем строковый вход в целочисленные индексы
    encoded_feature = lookup(inpt)

    return encoded_feature


def encode_inputs(inputs, input_names, entire_ds):
    """Предобрабатывает входные признаки.

    Для численного признака создаёт слой нормализации (вычитание среднего арифметического и
    масштабирование до [-1, 1]), а для категориального - слой one-hot-encoding.

    Args:
        inputs: список из входов нейросети.
        input_names: словарь {название_входа: его тип (численный, категориальный)}.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        encoded_features: список предобработанных признаков.
    """
    encoded_features = [
        encode_numerical_input(inpt, entire_ds)
        if defs.INPUT_NAMES[inpt.name] == 'numerical'
        else encode_categorical_input(inpt, entire_ds)
        for inpt in inputs
    ]

    return encoded_features


def create_model(layer_width, entire_ds):
    """Создаёт модель.

    Args:
        layer_width (int): ширина слоёв.
        entire_ds: целый датасет, из которого изучается статистика данных.

    Returns:
        model: собственно модель.
    """
    input_names = defs.INPUT_NAMES.copy()
    input_names.pop('label')

    inputs = create_inputs(input_names)
    encoded_inputs = encode_inputs(inputs, input_names, entire_ds)
    x = tf.keras.layers.concatenate(encoded_inputs)
    x = tf.keras.layers.Dense(layer_width, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(layer_width, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, output)

    return model
