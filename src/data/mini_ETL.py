import click
import numpy as np
import pandas as pd

import src


@click.command()
@click.argument('norm_data_path', type=click.Path(exists=True))
@click.argument('benign_tumor_data_path', type=click.Path(exists=True))
@click.argument('malignant_tumor_data_path', type=click.Path(exists=True))
@click.argument('etled_data_path', type=click.Path())
def mini_etl(
    norm_data_path: str,
    benign_tumor_data_path: str,
    malignant_tumor_data_path: str,
    etled_data_path: str,
) -> None:

    norm_data = pd.read_excel(
        norm_data_path, index_col=0, nrows=src.constants.NORM_N_ROWS)
    norm_data.drop(
        columns=['Дата рождения', 'Дата тестирования', 'Пол', '?1', '?2'], inplace=True)
    norm_data[src.constants.TARGET] = src.constants.LABELS[0]

    benign_tumor_data = pd.read_excel(
        benign_tumor_data_path, index_col=0, nrows=src.constants.BENIGN_TUMOR_N_ROWS)
    benign_tumor_data.drop(
        columns=['Дата рождения', 'Дата тестирования', '?1', '?2'], inplace=True)
    benign_tumor_data[src.constants.TARGET] = src.constants.LABELS[1]

    malignant_tumor_data = pd.read_excel(
        malignant_tumor_data_path,
        index_col=0,
        nrows=src.constants.MALIGNANT_TUMOR_N_ROWS,
    )
    malignant_tumor_data.drop(
        columns=['Дата рождения', 'Дата тестирования', 'Пол', '?1', '?2'], inplace=True)
    malignant_tumor_data[src.constants.TARGET] = src.constants.LABELS[2]

    data = pd.concat(
        [norm_data, benign_tumor_data, malignant_tumor_data], ignore_index=True)
    data.rename(
        columns={
            'Вопрос 2': src.constants.QUESTION_2,
            'Вопрос 3': src.constants.QUESTION_3,
            'Вопрос 4': src.constants.QUESTION_4,
            'Вопрос 5': src.constants.QUESTION_5,
            'Вопрос 6': src.constants.QUESTION_6,
            'Вопрос 7': src.constants.QUESTION_7,
            'Вопрос 8': src.constants.QUESTION_8,
            'Вопрос 9': src.constants.QUESTION_9,
            'Вопрос 10': src.constants.QUESTION_10,
            'Вопрос 11': src.constants.QUESTION_11,
            'Вопрос 12': src.constants.QUESTION_12,
            'Вопрос 13': src.constants.QUESTION_13,
            'Вопрос 14': src.constants.QUESTION_14,
            'Вопрос 15': src.constants.QUESTION_15,
            'Вопрос 16': src.constants.QUESTION_16,
            'Вопрос 17': src.constants.QUESTION_17,
            'Вопрос 18': src.constants.QUESTION_18,
            'Вопрос 19': src.constants.QUESTION_19,
            'Вопрос 20': src.constants.QUESTION_20,
            'Вопрос 22': src.constants.QUESTION_22,
            'Вопрос 23': src.constants.QUESTION_23,
            'Вопрос 24': src.constants.QUESTION_24,
            'Вопрос 25': src.constants.QUESTION_25,
            'Вопрос 26': src.constants.QUESTION_26,
            'Вопрос 27': src.constants.QUESTION_27,
            'Вопрос 28': src.constants.QUESTION_28,
            'Вопрос 29': src.constants.QUESTION_29,
            'Вопрос 30': src.constants.QUESTION_30,
            'Вопрос 31': src.constants.QUESTION_31,
            'Вопрос 32': src.constants.QUESTION_32,
            'Вопрос 33': src.constants.QUESTION_33,
            'Вопрос 34': src.constants.QUESTION_34,
            'Вопрос 35': src.constants.QUESTION_35,
        },
        inplace=True,
    )

    mask = (data[src.constants.QUESTION_2] > 85) | (data[src.constants.QUESTION_2] < 18)
    data.loc[mask, src.constants.QUESTION_2] = np.NaN

    data[src.constants.QUESTION_3].replace(
        to_replace={
            1: src.constants.ANSWER_3_1,
            2: src.constants.ANSWER_3_2,
            3: src.constants.ANSWER_3_3,
            4: src.constants.ANSWER_3_4,
            5: src.constants.ANSWER_3_5,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_4].replace(
        to_replace={
            '-': np.NaN,
            '?': np.NaN,
        },
        inplace=True,
    )
    data[src.constants.QUESTION_4] = data[src.constants.QUESTION_4].apply(convert_time_interval)

    data[src.constants.QUESTION_5].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_5_1,
            2: src.constants.ANSWER_5_2,
            3: src.constants.ANSWER_5_3,
            4: src.constants.ANSWER_5_4,
            5: src.constants.ANSWER_5_5,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_6].replace(
        to_replace={'нет ': src.constants.NO},
        inplace=True,
    )

    data[src.constants.QUESTION_7].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_7_1,
            2: src.constants.ANSWER_7_2,
            3: src.constants.ANSWER_7_3,
            4: src.constants.ANSWER_7_4,
            5: src.constants.ANSWER_7_5,
            6: src.constants.ANSWER_7_6,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_8].replace({
        ' нет': src.constants.NO,
        'да ': src.constants.YES,
        'нет ': src.constants.NO,
    }, inplace=True)

    data[src.constants.QUESTION_9].replace(
        to_replace={
            ' -': np.NaN,
            '-': np.NaN,
            0: np.NaN,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_10].replace(
        to_replace={
            '-': np.NaN,
            ' да': src.constants.YES,
            'да ': src.constants.YES,
            'дв': src.constants.YES,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_11].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_11_1,
            2: src.constants.ANSWER_11_2,
            3: src.constants.ANSWER_11_3,
            4: src.constants.ANSWER_11_4,
            5: src.constants.ANSWER_11_5,
            6: src.constants.ANSWER_11_6,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_12].replace(
        to_replace={
            1.: src.constants.ANSWER_12_1,
            2.: src.constants.ANSWER_12_2,
            3.: src.constants.ANSWER_12_3,
            4.: src.constants.ANSWER_12_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_13].replace(
        to_replace={
            1.: src.constants.ANSWER_13_1,
            2.: src.constants.ANSWER_13_2,
            3.: src.constants.ANSWER_13_3,
            33.: src.constants.ANSWER_13_3,
            4.: src.constants.ANSWER_13_4,
            5.: src.constants.ANSWER_13_5,
            6.: src.constants.ANSWER_13_6,
            7.: src.constants.ANSWER_13_7,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_14].replace(
        to_replace={
            1.: src.constants.ANSWER_14_1,
            2.: src.constants.ANSWER_14_2,
            3.: src.constants.ANSWER_14_3,
            4.: src.constants.ANSWER_14_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_15].replace(
        to_replace={
            '-': np.NaN,
            'да ': src.constants.YES,
            'ла': src.constants.YES,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_16].replace(to_replace={'-': np.NaN}, inplace=True)

    data[src.constants.QUESTION_17].replace(to_replace={'-': np.NaN}, inplace=True)

    data[src.constants.QUESTION_18].replace(to_replace={'-': np.NaN}, inplace=True)

    data[src.constants.QUESTION_19].replace(
        to_replace={
            1.: src.constants.ANSWER_19_20_1,
            2.: src.constants.ANSWER_19_20_2,
            3.: src.constants.ANSWER_19_20_3,
            4.: src.constants.ANSWER_19_20_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_20].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_19_20_1,
            2: src.constants.ANSWER_19_20_2,
            3: src.constants.ANSWER_19_20_3,
            4: src.constants.ANSWER_19_20_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_22].replace(
        to_replace={
            '-': np.NaN,
            '?': np.NaN,
        },
        inplace=True,
    )
    data[src.constants.QUESTION_22] = data[src.constants.QUESTION_22].apply(convert_time_interval)

    data[src.constants.QUESTION_23].replace(
        to_replace={
            1.: src.constants.ANSWER_23_1,
            2.: src.constants.ANSWER_23_2,
            3.: src.constants.ANSWER_23_3,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_24].replace(
        to_replace={
            '-': np.NaN,
            '?': np.NaN,
        },
        inplace=True,
    )
    data[src.constants.QUESTION_24] = data[src.constants.QUESTION_24].apply(convert_time_interval)

    data[src.constants.QUESTION_25].replace(
        to_replace={
            '-': np.NaN,
            1.: src.constants.ANSWER_25_1,
            2.: src.constants.ANSWER_25_2,
            3.: src.constants.ANSWER_25_3,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_26].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_26_1,
            2: src.constants.ANSWER_26_2,
            3: src.constants.ANSWER_26_3,
            4: np.NaN,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_27].replace(
        to_replace={
            '-': np.NaN,
            1: src.constants.ANSWER_27_1,
            2: src.constants.ANSWER_27_2,
            3: src.constants.ANSWER_27_3,
            4: src.constants.ANSWER_27_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_28].replace(
        to_replace={
            1.: src.constants.ANSWER_28_1,
            2.: src.constants.ANSWER_28_2,
            3.: src.constants.ANSWER_28_3,
            4.: src.constants.ANSWER_28_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_29].replace(
        to_replace={
            1.: src.constants.ANSWER_29_1,
            2.: src.constants.ANSWER_29_2,
            3.: src.constants.ANSWER_29_3,
            4.: src.constants.ANSWER_29_4,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_30].replace(
        to_replace={
            'не т': src.constants.NO,
            'нет ': src.constants.NO,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_33].replace(
        to_replace={
            'НЕТ': src.constants.NO,
            'да ': src.constants.YES,
            'есть': src.constants.YES,
            'есть ': src.constants.YES,
            'ней': src.constants.NO,
            'нте': src.constants.NO,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_34].replace(
        to_replace={
            'НЕТ': src.constants.NO,
            'дв': src.constants.YES,
            'есть': src.constants.YES,
            'есть ': src.constants.YES,
            'нет ': src.constants.NO,
        },
        inplace=True,
    )

    data[src.constants.QUESTION_35].replace(
        to_replace={
            ' нет': src.constants.NO,
            'есть': src.constants.YES,
        },
        inplace=True,
    )

    data.to_csv(etled_data_path)


def remove_no_digits(string: str) -> str:
    """Удаляет из строки нецифры."""
    digits = ''
    for char in string:
        if char.isdigit() or char == '.':
            digits += char

    return digits


def convert_time_interval(cell_content) -> float:
    """Конвертирует временной интервал из строки в float формат в годах."""
    if isinstance(cell_content, str):
        result = float(remove_no_digits(cell_content))
        if 'месяц' in cell_content:
            result /= 12
    else:
        result = cell_content

    return result


if __name__ == '__main__':
    mini_etl()
