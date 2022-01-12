"""Здесь содержатся всякие определения, которые не хочется переносить из файла в файл."""
import os

import tensorflow as tf


YES = 'да'
NO = 'нет'


QUESTION_2 = '2. Возраст'

QUESTION_3 = '3. Семейное положение'
ANSWER_3_1 = 'замужем'
ANSWER_3_2 = 'имею гражданского супруга'
ANSWER_3_3 = 'одинока'
ANSWER_3_4 = 'разведена'
ANSWER_3_5 = 'вдова'
ANSWERS_3 = [
    ANSWER_3_1,
    ANSWER_3_2,
    ANSWER_3_3,
    ANSWER_3_4,
    ANSWER_3_5
]

QUESTION_4 = '4. Если имеете супруга или партнера, как долго вы живете вместе (в годах)?'

QUESTION_5 = '5. В какой семье Вы выросли?'
ANSWER_5_1 = 'полная семья, кровные родители'
ANSWER_5_2 = 'мачеха/отчим'
ANSWER_5_3 = 'мать/отец одиночка'
ANSWER_5_4 = 'с бабушкой и дедушкой'
ANSWER_5_5 = 'в детском доме'
ANSWERS_5 = [
    ANSWER_5_1,
    ANSWER_5_2,
    ANSWER_5_3,
    ANSWER_5_4,
    ANSWER_5_5
]

QUESTION_6 = '6. Жив ли хотя бы один из Ваших родителей (да/нет)?'

QUESTION_7 = '7. Если да, как часто вы общаетесь?'
ANSWER_7_1 = 'я живу с моими родителями'
ANSWER_7_2 = 'каждый день или почти каждый день'
ANSWER_7_3 = 'раз в неделю'
ANSWER_7_4 = 'один-два раза в месяц'
ANSWER_7_5 = 'несколько раз в год'
ANSWER_7_6 = 'я не общаюсь с родителями'
ANSWERS_7 = [
    ANSWER_7_1,
    ANSWER_7_2,
    ANSWER_7_3,
    ANSWER_7_4,
    ANSWER_7_5,
    ANSWER_7_6
]

QUESTION_8 = '8. Есть ли у Вас дети (да/нет)?'

QUESTION_9 = '9. Если да, сколько?'
ANSWERS_9 = [0., 1., 2., 3., 4., 5., 6.]

QUESTION_10 = '10. Есть ли у Вас совершеннолетние дети (да/нет)?'

QUESTION_11 = '11. Если да, как часто вы общаетесь?'
ANSWER_11_1 = 'я живу с моими взрослыми детьми'
ANSWER_11_2 = 'каждый день или почти каждый день'
ANSWER_11_3 = 'раз в неделю'
ANSWER_11_4 = 'один-два раза в месяц'
ANSWER_11_5 = 'несколько раз в год'
ANSWER_11_6 = 'я не общаюсь со взрослыми детьми'
ANSWERS_11 = [
    ANSWER_11_1,
    ANSWER_11_2,
    ANSWER_11_3,
    ANSWER_11_4,
    ANSWER_11_5,
    ANSWER_11_6
]

QUESTION_12 = '12. Сколько человек живут вместе с Вами?'
ANSWER_12_1 = 'я живу одна'
ANSWER_12_2 = '1 человек'
ANSWER_12_3 = '2-3 человека'
ANSWER_12_4 = '4 и более человек'
ANSWERS_12 = [
    ANSWER_12_1,
    ANSWER_12_2,
    ANSWER_12_3,
    ANSWER_12_4
]

QUESTION_13 = '13. Каковы Ваши взаимоотношения с соседями?'
ANSWER_13_1 = 'очень хорошие, дружеские'
ANSWER_13_2 = 'хорошие'
ANSWER_13_3 = 'нейтральные'
ANSWER_13_4 = 'скорее плохие'
ANSWER_13_5 = 'очень плохие'
ANSWER_13_6 = 'я не знаю своих соседей'
ANSWER_13_7 = 'у меня нет соседей'
ANSWERS_13 = [
    ANSWER_13_1,
    ANSWER_13_2,
    ANSWER_13_3,
    ANSWER_13_4,
    ANSWER_13_5,
    ANSWER_13_6,
    ANSWER_13_7
]

QUESTION_14 = '14. Как часто Вы встречаетесь с друзьями?'
ANSWER_14_1 = 'несколько раз в неделю'
ANSWER_14_2 = 'раз в неделю'
ANSWER_14_3 = 'раз в месяц'
ANSWER_14_4 = 'реже, чем раз в месяц'
ANSWERS_14 = [
    ANSWER_14_1,
    ANSWER_14_2,
    ANSWER_14_3,
    ANSWER_14_4
]

QUESTION_15 = '15. Есть ли у Вас домашние питомцы (да/нет)?'

QUESTION_16 = '16. В течение последних 7 дней, как часто Вы практиковали тяжелые физические нагрузки?'
QUESTION_17 = '17. В течение последних 7 дней, как часто Вы практиковали умеренные физические нагрузки?'
QUESTION_18 = '18. В течение последних 7 дней, как часто Вы ходили пешком минимум 10 минут без перерыва?'
ANSWERS_16_17_18 = [0., 1., 2., 3., 4., 5., 6., 7.]

QUESTION_19 = '19. Уровень Вашего образования?'
QUESTION_20 = '20. Каков уровень образования Вашего партнера (если применимо)?'
ANSWER_19_20_1 = 'средняя школа'
ANSWER_19_20_2 = 'среднее специальное образование'
ANSWER_19_20_3 = 'законченное высшее образование'
ANSWER_19_20_4 = 'учёная степень'
ANSWERS_19_20 = [
    ANSWER_19_20_1,
    ANSWER_19_20_2,
    ANSWER_19_20_3,
    ANSWER_19_20_4
]

QUESTION_22 = '22. Как долго Вы проживаете в этом месте (в годах)?'

QUESTION_23 = '23. Каков тип Вашего дома?'
ANSWER_23_1 = 'многоквартирный дом'
ANSWER_23_2 = 'таунхаус'
ANSWER_23_3 = 'собственный дом'
ANSWERS_23 = [
    ANSWER_23_1,
    ANSWER_23_2,
    ANSWER_23_3
]

QUESTION_24 = '24. Если Вы живете в многоквартирном доме, то на каком этаже?'

QUESTION_25 = '25. Каким транспортом Вы обычно пользуетесь?'
ANSWER_25_1 = 'общественный транспорт'
ANSWER_25_2 = 'собственная машина/такси'
ANSWER_25_3 = 'я обычно не пользуюсь транспортом'
ANSWERS_25 = [
    ANSWER_25_1,
    ANSWER_25_2,
    ANSWER_25_3
]

QUESTION_26 = '26. Сколько времени занимает Ваш путь до работы в одну сторону?'
ANSWER_26_1 = '1 час и меньше'
ANSWER_26_2 = '1-3 часа'
ANSWER_26_3 = 'более трёх часов'
ANSWERS_26 = [
    ANSWER_26_1,
    ANSWER_26_2,
    ANSWER_26_3
]

QUESTION_27 = '27. Каков тип Вашей занятости?'
ANSWER_27_1 = 'полный рабочий день'
ANSWER_27_2 = 'частичная занятость'
ANSWER_27_3 = 'я работаю из дома'
ANSWER_27_4 = 'я не работаю'
ANSWERS_27 = [
    ANSWER_27_1,
    ANSWER_27_2,
    ANSWER_27_3,
    ANSWER_27_4
]

QUESTION_28 = '28. Каковы Ваши предпочтения в пище?'
ANSWER_28_1 = 'я ем мясо или рыбу ежедневно'
ANSWER_28_2 = 'я ем мясо или рыбу 2-3 раза в неделю'
ANSWER_28_3 = 'я практически не ем мясо/рыбу'
ANSWER_28_4 = 'я вегетарианец/веган'
ANSWERS_28 = [
    ANSWER_28_1,
    ANSWER_28_2,
    ANSWER_28_3,
    ANSWER_28_4
]

QUESTION_29 = '29. Каков тип Вашего питания?'
ANSWER_29_1 = '3-4-разовое домашнее питание'
ANSWER_29_2 = '3-разовое питание, домашнее и в предприятиях общественного питания'
ANSWER_29_3 = 'дома готовлю редко, питаюсь в предприятиях общественного питания'
ANSWER_29_4 = 'регулярного режима питания нет'
ANSWERS_29 = [
    ANSWER_29_1,
    ANSWER_29_2,
    ANSWER_29_3,
    ANSWER_29_4
]

QUESTION_30 = '30. Вы курите (да/нет)?'

QUESTION_31 = '31. Количество родов'
ANSWERS_31 = [0., 1., 2., 3., 4., 5., 6.]

QUESTION_32 = '32. Количество прерванных беременностей'
ANSWERS_32 = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]

QUESTION_33 = '33. Гинекологические заболевания (да/нет)'

QUESTION_34 = '34. Заболевания щитовидной железы (да/нет)'

QUESTION_35 = '35. Наследственность (да/нет)'

LABEL = 'Метка'
LABEL_1 = 'норма'
LABEL_2 = 'доброкачественная опухоль'
LABEL_3 = 'злокачественная опухоль'
LABELS = [
    LABEL_1,
    LABEL_2,
    LABEL_3
]


COLUMN_NAMES = {
    'Вопрос 2': QUESTION_2,
    'Вопрос 3': QUESTION_3,
    'Вопрос 4': QUESTION_4,
    'Вопрос 5': QUESTION_5,
    'Вопрос 6': QUESTION_6,
    'Вопрос 7': QUESTION_7,
    'Вопрос 8': QUESTION_8,
    'Вопрос 9': QUESTION_9,
    'Вопрос 10': QUESTION_10,
    'Вопрос 11': QUESTION_11,
    'Вопрос 12': QUESTION_12,
    'Вопрос 13': QUESTION_13,
    'Вопрос 14': QUESTION_14,
    'Вопрос 15': QUESTION_15,
    'Вопрос 16': QUESTION_16,
    'Вопрос 17': QUESTION_17,
    'Вопрос 18': QUESTION_18,
    'Вопрос 19': QUESTION_19,
    'Вопрос 20': QUESTION_20,
    'Вопрос 22': QUESTION_22,
    'Вопрос 23': QUESTION_23,
    'Вопрос 24': QUESTION_24,
    'Вопрос 25': QUESTION_25,
    'Вопрос 26': QUESTION_26,
    'Вопрос 27': QUESTION_27,
    'Вопрос 28': QUESTION_28,
    'Вопрос 29': QUESTION_29,
    'Вопрос 30': QUESTION_30,
    'Вопрос 31': QUESTION_31,
    'Вопрос 32': QUESTION_32,
    'Вопрос 33': QUESTION_33,
    'Вопрос 34': QUESTION_34,
    'Вопрос 35': QUESTION_35,
}


CATEGORICAL_COLUMN_NAMES = {
    # QUESTION_1 - отсутствует
    # QUESTION_2 - численный
    QUESTION_3: ANSWERS_3,
    # QUESTION_4 - численный
    QUESTION_5: ANSWERS_5,
    QUESTION_6: [YES, NO],
    QUESTION_7: ANSWERS_7,
    # QUESTION_8 - включён в 9
    QUESTION_9: ANSWERS_9,
    QUESTION_10: [YES, NO],
    QUESTION_11: ANSWERS_11,
    QUESTION_12: ANSWERS_12,
    QUESTION_13: ANSWERS_13,
    QUESTION_14: ANSWERS_14,
    QUESTION_15: [YES, NO],
    QUESTION_16: ANSWERS_16_17_18,
    QUESTION_17: ANSWERS_16_17_18,
    QUESTION_18: ANSWERS_16_17_18,
    QUESTION_19: ANSWERS_19_20,
    QUESTION_20: ANSWERS_19_20,
    # QUESTION_21 - отсутствует
    # QUESTION_22 - численный
    QUESTION_23: ANSWERS_23,
    # QUESTION_24 - численный
    QUESTION_25: ANSWERS_25,
    QUESTION_26: ANSWERS_26,
    QUESTION_27: ANSWERS_27,
    QUESTION_28: ANSWERS_28,
    QUESTION_29: ANSWERS_29,
    QUESTION_30: [YES, NO],
    QUESTION_31: ANSWERS_31,
    QUESTION_32: ANSWERS_32,
    QUESTION_33: [YES, NO],
    QUESTION_34: [YES, NO],
    QUESTION_35: [YES, NO],
    LABEL: LABELS
}


INPUT_NAMES = {
                                # Имя
    'input_2': 'numerical',     # Возраст
    'input_3': 'categorical',   # Семейное положение
    'input_4': 'numerical',     # Если имеете супруга или партнера, как долго вы живете вместе (в годах)?
    'input_5': 'categorical',   # В какой семье Вы выросли?
    'input_6': 'categorical',   # Жив ли хотя бы один из Ваших родителей (да/нет)?
    'input_7': 'categorical',   # Если да, как часто вы общаетесь?
                                # Есть ли у Вас дети (да/нет)?
    'input_9': 'categorical',   # Если да, сколько?
    'input_10': 'categorical',  # Есть ли у Вас совершеннолетние дети (да/нет)?
    'input_11': 'categorical',  # Если да, как часто вы общаетесь?
    'input_12': 'categorical',  # Сколько человек живут вместе с Вами?
    'input_13': 'categorical',  # Каковы Ваши взаимоотношения с соседями?
    'input_14': 'categorical',  # Как часто Вы встречаетесь с друзьями?
    'input_15': 'categorical',  # Есть ли у Вас домашние питомцы (да/нет)?
    'input_16': 'categorical',  # В течение последних 7 дней, как часто Вы практиковали тяжелые физические нагрузки?
    'input_17': 'categorical',  # В течение последних 7 дней, как часто Вы практиковали умеренные физические нагрузки?
    'input_18': 'categorical',  # В течение последних 7 дней, как часто Вы ходили пешком минимум 10 минут без перерыва?
    'input_19': 'categorical',  # Уровень Вашего образования?
    'input_20': 'categorical',  # Каков уровень образования Вашего партнера (если применимо)?
                                # Пропущенный вопрос
    'input_22': 'numerical',    # Как долго Вы проживаете в этом месте (в годах)?
    'input_23': 'categorical',  # Каков тип Вашего дома?
    'input_24': 'numerical',    # Если Вы живете в многоквартирном доме, то на каком этаже?
    'input_25': 'categorical',  # Каким транспортом Вы обычно пользуетесь?
    'input_26': 'categorical',  # Сколько времени занимает Ваш путь до работы в одну сторону?
    'input_27': 'categorical',  # Каков тип Вашей занятости?
    'input_28': 'categorical',  # Каковы Ваши предпочтения в пище?
    'input_29': 'categorical',  # Каков тип Вашего питания?
    'input_30': 'categorical',  # Вы курите (да/нет)?
    'input_31': 'categorical',  # Количество родов
    'input_32': 'categorical',  # Количество прерванных беременностей
    'input_33': 'categorical',  # Гинекологические заболевания (да/нет)
    'input_34': 'categorical',  # Заболевания щитовидной железы (да/нет)
    'input_35': 'categorical',  # Наследственность (да/нет)
    'label': None
}

PREPARED_DATA_PATH = os.path.join('data', 'prepared data.xlsx')
GENERALIZING_MODEL = os.path.join(os.getcwd(), 'models', 'generalizing model')
OVERFITTED_MODEL = os.path.join(os.getcwd(), 'models', 'overfitted model')

ENTIRE_DATA_PATH = os.path.join('data', 'entire data.csv')
TRAIN_DATA_PATH = os.path.join('data', 'train data.csv')
VALIDATION_DATA_PATH = os.path.join('data', 'validation data.csv')
TEST_DATA_PATH = os.path.join('data', 'test data.csv')


def get_dataset_from_csv(file_path):
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
        column_names=INPUT_NAMES.keys(),
        label_name='label',
        header=False,
        num_epochs=1
    )
    # one-hot-encoding метки
    dataset = dataset.map(lambda point, label: (point, tf.one_hot(label, depth=3)))
    dataset.batch(1)

    return dataset
