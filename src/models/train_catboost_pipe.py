import catboost
import click
import joblib
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

import src


CAT_FEATURES = sorted(list(src.constants.CATEGORICAL_FEATURES))
FEATURES_TO_IMPUT = [src.constants.QUESTION_2]
USEFUL_FOR_IMPUT = [
    src.constants.QUESTION_6, src.constants.QUESTION_8, src.constants.QUESTION_31,
    src.constants.QUESTION_32, src.constants.QUESTION_33, src.constants.QUESTION_34,
    src.constants.QUESTION_35,
]


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_catboost_pipe(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path, index_col=0)

    X_train = train_data.drop(columns=src.constants.TARGET)
    y_train = train_data[src.constants.TARGET]

    model = get_model()
    model.fit(X_train, y_train)

    joblib.dump(model, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:

    constant_imputer = SimpleImputer(strategy='constant', fill_value='пропуск')

    preprocessing = ColumnTransformer(
        transformers=[
            ('knn_imputer', KNNImputer(), FEATURES_TO_IMPUT + USEFUL_FOR_IMPUT),
            ('simple_imputer', constant_imputer, CAT_FEATURES),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    catboost_model = catboost.CatBoostClassifier(
        grow_policy='Lossguide',
        n_estimators=25,
        learning_rate=.05,
        max_depth=5,
        max_leaves=15,
        min_data_in_leaf=.10966755590229849,
        cat_features=CAT_FEATURES,
        verbose=False,
        random_state=src.constants.RANDOM_STATE,
    )

    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('catboost_model', catboost_model),
    ])

    return pipe


if __name__ == '__main__':
    train_catboost_pipe()
