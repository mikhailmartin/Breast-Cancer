"""Здесь реализован скрипт, разбивающий датасет на обучающую, валидационную и тестовую части."""
import os

import numpy as np
import pandas as pd

import definitions as defs


def train_val_test_split(df, test_frac=.1, val_frac=.1, seed=None, verbose=False):
    """Разбивает целый DataFrame на обучающую, валидационную и тестовую части.

    Args:
        df: целый Dataframe.
        test_frac (float): доля тестовой части.
        val_frac (float): доля валидационной части.
        seed: для воспроизводимости результатов.
        verbose (bool): печатать ли сведения о количестве точек данных в частях.

    Returns:
        train_part (DataFrame): обучающая часть.
        val_part (DataFrame): валидационная часть.
        test_part (DataFrame): тестовая часть.
    """
    # создаём рандом генератор
    rng = np.random.default_rng(seed)
    # перетасовываем индексы
    perm = rng.permutation(len(df.index))

    size = len(perm)
    test_end = int(test_frac * size)
    val_end = int(val_frac * size) + test_end

    test_part = df.iloc[perm[:test_end]]
    val_part = df.iloc[perm[test_end:val_end]]
    train_part = df.iloc[perm[val_end:]]
    # небольшая справка о количестве точек данных в каждой из частей
    if verbose:
        print(f'Обучающая часть содержит {len(train_part)} точек данных.')
        print(f'Валидационная часть содержит {len(val_part)} точек данных.')
        print(f'Тестовая часть содержит {len(test_part)} точек данных.')

    return train_part, val_part, test_part


if __name__ == '__main__':
    # читаем DataFrame из EXCEL-файла
    dataframe = pd.read_excel(os.path.join('..', defs.ETLED_DATA_PATH))
    # разбиваем DataFrame на части
    train_dataframe, val_dataframe, test_dataframe = train_val_test_split(
        dataframe, test_frac=.1, val_frac=.1, seed=1234)
    # сохраняем части в CSV-файлы
    train_dataframe.to_csv(
        path_or_buf=os.path.join('..', defs.TRAIN_DATA_PATH),
        header=False,
        index=False)
    val_dataframe.to_csv(
        path_or_buf=os.path.join('..', defs.VAL_DATA_PATH),
        header=False,
        index=False)
    test_dataframe.to_csv(
        path_or_buf=os.path.join('..', defs.TEST_DATA_PATH),
        header=False,
        index=False)

    # сохраняем и весь DataFrame в CSV-файл
    dataframe.to_csv(
        path_or_buf=os.path.join('..', defs.ENTIRE_DATA_PATH),
        header=False,
        index=False
    )
