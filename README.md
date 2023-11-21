# Оценка значимости признаков для задачи классификации на примере данных о новообразованиях молочной железы

Этот проект был реализован в рамках выполнения ВКР (выпускной квалификационной работы)
магистра и научной работы в лаборатории управления здравоохранением при Институте Проблем
Управления им. В.А. Трапезникова РАН.


## Набор данных
Датасет был предоставлен в ИПУ. Из себя он представляет 595 заполненных анкет-опросников,
размеченных на 3 класса: «норма», «доброкачественная опухоль» и «злокачественная
опухоль». Каждый по чуть меньше 200 анкет. Как вы могли догадаться, их заполняли женщины,
которые проходили обследование на наличие и характер новообразований молочной железы.

<p align="center">
  <img src="https://raw.githubusercontent.com/mikhailmartin/Breast-Cancer/master/reports/figures/target_distribution.png"/>
</p>


## Структура проекта

старается соответствовать шаблону [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science).

```
├── README.md
|
├── data/
|   ├── raw/                                          <- Исходные Excel-таблички
|   ├── interim/                                      <- Собранный воедино и почищенный в ходе ETL датасет
|   └── processed/                                    <- Осмысленное OrdinalEncoding
|
├── models/                                           <- Обученные модели-классификаторы
|
├── notebooks/
|   ├── EDA.ipynb                                     <- EDA с визуализацией разниц в распределении признаков
|   |                                                    между классами. Также здесь исследую примеры с
|   |                                                    противоречивыми признаками.
|   ├── model_DecisionTreeClassifier.ipynb            <- Эксперименты с деревом решений из scikit-learn
|   ├── model_MultiSplitDecisionTreeCalssifier.ipynb  <- Эксперименты с собственным деревом решений с
|   |                                                    мультисплитами
|   ├── model_NeuralNetwork.ipynb                     <- Эксперименты с персептроном-классификатором с двумя
|   |                                                    скрытыми слоями
|   ├── model_CatBoost.ipynb                          <- Эксперименты с GBDT CatBoost
|   └── model_LightGBM.ipynb                          <- Эксперименты с GBDT LightGBM
|
├── reports/
|   └── figures/
|
├── win_requirements.txt                              <- Необходимые пакеты для виртуального окружения
|
├── my_ds_tools/                                      <- Сабмодуль с тулзами
|
└── src/
    ├── __init__.py
    |
    ├── constants.py                                  <- Константы
    |
    ├── data/
    |   ├── __init__.py
    |   ├── mini_ETL.py                               <- ETL на минималках: свожу excel-таблицы в единый датасет,
    |   |                                                расставляю целевые переменные и исправляю опечатки
    |   └── meaningful_ordinal_encoding.py            <- Осмысленный OrdinalEncoding ранговых признаков
    |
    ├── models/                                       <- Скрипты обучения моделей
    |   ├── __init__.py
    |   ├── train_catboost_pipe.py
    |   ├── train_lightgbm_pipe.py
    |   ├── train_decision_tree_pipe.py
    |   └── train_my_decision_tree_pipe.py
    |
    └── visualization/
        ├── __init__.py
        └── visualize.py
```


## Результаты экспериментов
| Модель                              | LeaveOneOut cross-validation accuracy | train accuracy   | test accuracy    |
|-------------------------------------|---------------------------------------|------------------|------------------|
| SciKit-Learn DecisionTreeClassifier | 0.59                                  | 0.60             | 0.53             |
| MultiSplitDecisionTreeCalssifier    | (на переработке)                      | (на переработке) | (на переработке) |
| GBDT CatBoost                       | 0.61                                  | 0.69             | 0.56             |
| GBDT LightGBM                       | 0.64                                  | 0.66             | 0.52             |
| Neural Network                      | (на переработке)                      | (на переработке) | (на переработке) |

Модель на CatBoost была выбрана в качестве базового классификатора. Значимость признаков
оценивалась через SHAP значения.

<p align="center">
  <img src="https://raw.githubusercontent.com/mikhailmartin/Breast-Cancer/master/reports/figures/SHAP_values.png"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mikhailmartin/Breast-Cancer/master/reports/figures/Permutation_Importance.png"/>
</p>


## Интерпретатор, окружение, WorkFlow
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

Подтягиваем сабмодуль
```commandline
git submodule init
git submodule update
```

Прогоняем WorkFlow:
```commandline
dvc repro
```
