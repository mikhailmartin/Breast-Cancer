import catboost
import joblib
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

import src


FEATURES_TO_IMPUT = [src.constants.QUESTION_2]
USEFUL_FOR_IMPUT = [
    src.constants.QUESTION_6, src.constants.QUESTION_8, src.constants.QUESTION_31,
    src.constants.QUESTION_32, src.constants.QUESTION_33, src.constants.QUESTION_34,
    src.constants.QUESTION_35,
]


def train_catboost(input_data_path: str, output_model_path: str) -> None:
    data = pd.read_csv(input_data_path)
    X = data.drop(columns=src.constants.TARGET)
    y = data[src.constants.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=src.constants.RANDOM_STATE)

    model = get_model()
    model.fit(X_train, y_train)

    joblib.dump(model, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ('imputer', KNNImputer(), FEATURES_TO_IMPUT + USEFUL_FOR_IMPUT),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    catboost_model = catboost.CatBoostClassifier(
        random_state=src.constants.RANDOM_STATE,
    )

    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('catboost_model', catboost_model),
    ])

    return pipe


if __name__ == '__main__':
    train_catboost(src.constants.ENCODED_DATA_PATH, src.constants.CATBOOST_MODEL_PATH)
