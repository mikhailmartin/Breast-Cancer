"""Здесь содержатся всякие определения, которые не хочется переносить из файла в файл."""
import os


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
    ANSWER_3_5,
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
    ANSWER_5_5,
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
    ANSWER_7_6,
]

QUESTION_8 = '8. Есть ли у Вас дети (да/нет)?'

QUESTION_9 = '9. Если да, сколько?'
ANSWERS_9 = [
    0.,
    1.,
    2.,
    3.,
    4.,
    5.,
    6.,
]

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
    ANSWER_11_6,
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
    ANSWER_12_4,
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
    ANSWER_13_7,
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
    ANSWER_14_4,
]

QUESTION_15 = '15. Есть ли у Вас домашние питомцы (да/нет)?'

QUESTION_16 = ('16. В течение последних 7 дней, как часто Вы практиковали тяжелые физические '
               'нагрузки?')
QUESTION_17 = ('17. В течение последних 7 дней, как часто Вы практиковали умеренные физические '
               'нагрузки?')
QUESTION_18 = ('18. В течение последних 7 дней, как часто Вы ходили пешком минимум 10 минут без '
               'перерыва?')
ANSWERS_16_17_18 = [
    0.,
    1.,
    2.,
    3.,
    4.,
    5.,
    6.,
    7.,
]

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
    ANSWER_19_20_4,
]

QUESTION_22 = '22. Как долго Вы проживаете в этом месте (в годах)?'

QUESTION_23 = '23. Каков тип Вашего дома?'
ANSWER_23_1 = 'многоквартирный дом'
ANSWER_23_2 = 'таунхаус'
ANSWER_23_3 = 'собственный дом'
ANSWERS_23 = [
    ANSWER_23_1,
    ANSWER_23_2,
    ANSWER_23_3,
]

QUESTION_24 = '24. Если Вы живете в многоквартирном доме, то на каком этаже?'

QUESTION_25 = '25. Каким транспортом Вы обычно пользуетесь?'
ANSWER_25_1 = 'общественный транспорт'
ANSWER_25_2 = 'собственная машина/такси'
ANSWER_25_3 = 'я обычно не пользуюсь транспортом'
ANSWERS_25 = [
    ANSWER_25_1,
    ANSWER_25_2,
    ANSWER_25_3,
]

QUESTION_26 = '26. Сколько времени занимает Ваш путь до работы в одну сторону?'
ANSWER_26_1 = '1 час и меньше'
ANSWER_26_2 = '1-3 часа'
ANSWER_26_3 = 'более трёх часов'
ANSWERS_26 = [
    ANSWER_26_1,
    ANSWER_26_2,
    ANSWER_26_3,
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
    ANSWER_27_4,
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
    ANSWER_28_4,
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
    ANSWER_29_4,
]

QUESTION_30 = '30. Вы курите (да/нет)?'

QUESTION_31 = '31. Количество родов'
ANSWERS_31 = [
    0.,
    1.,
    2.,
    3.,
    4.,
    5.,
    6.,
]

QUESTION_32 = '32. Количество прерванных беременностей'
ANSWERS_32 = [
    0.,
    1.,
    2.,
    3.,
    4.,
    5.,
    6.,
    7.,
    8.,
    9.,
    10.,
    11.,
    12.,
]

QUESTION_33 = '33. Гинекологические заболевания (да/нет)'

QUESTION_34 = '34. Заболевания щитовидной железы (да/нет)'

QUESTION_35 = '35. Наследственность (да/нет)'

TARGET = 'Метка'
LABEL_1 = 'норма'
LABEL_2 = 'доброкачественная опухоль'
LABEL_3 = 'злокачественная опухоль'
LABELS = [
    LABEL_1,
    LABEL_2,
    LABEL_3,
]

ETLED_DATA_PATH = os.path.join('data', 'ETLed data.xlsx')
NEURAL_NETWORK = os.path.join(os.getcwd(), 'models', 'neural network')
