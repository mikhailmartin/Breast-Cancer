import catboost
import click
import joblib
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

import src


FEATURES_TO_IMPUT = [src.constants.QUESTION_2]
USEFUL_FOR_IMPUT = [
    src.constants.QUESTION_6, src.constants.QUESTION_8, src.constants.QUESTION_31,
    src.constants.QUESTION_32, src.constants.QUESTION_33, src.constants.QUESTION_34,
    src.constants.QUESTION_35,
]


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_catboost(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path)

    X_train = train_data.drop(columns=src.constants.TARGET)
    y_train = train_data[src.constants.TARGET]

    model = get_model()
    model.fit(X_train, y_train)

    joblib.dump(model, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:

    knn_imputer = KNNImputer()
    simple_imputer = SimpleImputer(strategy='constant', fill_value='пропуск')

    preprocessing = ColumnTransformer(
        transformers=[
            ('knn_imputer', knn_imputer, FEATURES_TO_IMPUT + USEFUL_FOR_IMPUT),
            ('simple_imputer', simple_imputer, list(src.constants.CATEGORICAL_FEATURES)),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    catboost_model = catboost.CatBoostClassifier(
        max_depth=5,
        cat_features=list(src.constants.CATEGORICAL_FEATURES),
        verbose=False,
        random_state=src.constants.RANDOM_STATE,
    )

    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('catboost_model', catboost_model),
    ])

    return pipe


if __name__ == '__main__':
    train_catboost()
