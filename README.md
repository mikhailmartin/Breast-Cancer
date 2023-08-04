# Оценка значимости признаков для задачи классификации на примере данных о новообразованиях молочной железы

Этот проект был реализован в рамках выполнения ВКР (выпускной квалификационной работы) магистра и
научной работы в лаборатории управления здравоохранением при Институте Проблем Управления им. В.А.
Трапезникова РАН.


## Набор данных
Датасет был предоставлен в ИПУ. Из себя он представляет 595 заполненных анкет-опросников,
размеченных на 3 класса: «норма», «доброкачественная опухоль» и «злокачественная опухоль». Каждый
по чуть меньше 200 анкет. Как вы могли догадаться, их заполняли женщины, которые проходили
обследование на наличие и характер новообразований молочной железы.

<p align="center">
  <img src="https://raw.githubusercontent.com/mikhailmartin/Breast-Cancer/master/reports/figures/target distribution.png"/>
</p>


## Структура проекта

старается соответствовать шаблону [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science).

```
├── README.md
├── data
|   ├── raw                              <- Исходные Excel-таблички
|   ├── interim                          <- Собранный воедино и почищенный в ходе ETL датасет
|   └── processed                        <- Осмысленное OrdinalEncoding
|
├── models                               <- Обученные модели-классификаторы
|
├── notebooks
|   ├── EDA.ipynb                        <- EDA с визуализацией разниц в распределении признаков
|   |                                       между классами. Также здесь исследую примеры с
|   |                                       противоречивыми признаками.
|   ├── DecisionTreeClassifier.ipynb     <- Эксперименты с деревом решений из scikit-learn
|   ├── MyDecisionTreeClassifier.ipynb   <- Эксперименты с собственным деревом решений с
|   |                                       мультисплитами
|   ├── Neural Network.ipynb             <- Эксперименты с персептроном-классификатором с двумя
|   |                                       скрытыми слоями
|   └── CatBoost.ipynb                   <- Эксперименты с GBDT CatBoost
|
├── reports
|   └── figures
|
├── win_requirements.txt                 <- Необходимые пакеты для виртуального окружения
|
├── my_ds_tools                          <- Сабмодуль с тулзами
|
└── src
    ├── __init__.py
    |
    ├── constants.py                     <- Константы
    |
    ├── decision_tree.py                 <- Класс вышеупомянутого собственного дерева решений
    |
    ├── data
    |   ├── __init__.py
    |   ├── mini_ETL.py                  <- ETL на минималках: свожу excel-таблицы в единый датасет,
    |   |                                   расставляю целевые переменные и исправляю опечатки
    |   └── meaningful_ordinal_encoding.py  <- Осмысленный OrdinalEncoding ранговых признаков
    |
    ├── models                           <- Скрипты обучения моделей
    |   ├── __init__.py
    |   ├── train_catboost_pipe.py
    |   ├── train_decision_tree_pipe.py
    |   └── train_my_decision_tree_pipe.py
    |
    └── visualization
        ├── __init__.py
        └── visualize.py
```


## Результаты экспериментов
| Модель                         | mean CV accuracy   |
|--------------------------------|--------------------|
| sklearn DecisionTreeClassifier | 0.5942222222222223 |
| MyDecisionTreeClassifier       | 0.5466666666666666 |
| Neural Network                 |                    |
| CatBoost                       | 0.6977777777777778 |

Модель на CatBoost была выбрана в качестве базового классификатора. Значимость признаков оценивалась
через SHAP значения.

<p align="center">
  <img src="https://raw.githubusercontent.com/mikhailmartin/Breast-Cancer/master/reports/figures/SHAP_values.png"/>
</p>


## Интерпретатор и окружение
Я использовал интерпретатор [Python 3.11.4](https://www.python.org/downloads/release/python-3114/).

Создаём и активируем виртуальное окружение:
```commandline
python -m venv env
call ./venv/scripts/activate
```

Устанавливаем все необходимые пакеты, собранные в `win_requirements.txt`:
```commandline
pip install -r ./win_requirements.txt
```

Прогоняем WorkFlow:
```commandline
dvc repro
```
