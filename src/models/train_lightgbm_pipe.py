import click
import joblib
import lightgbm
import pandas as pd

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

import src


CAT_FEATURES = sorted(list(src.constants.CATEGORICAL_FEATURES))
FEATURES_TO_IMPUT = [src.constants.QUESTION_2]
USEFUL_FOR_IMPUT = [
    src.constants.QUESTION_6, src.constants.QUESTION_8, src.constants.QUESTION_31,
    src.constants.QUESTION_32, src.constants.QUESTION_33, src.constants.QUESTION_34,
    src.constants.QUESTION_35,
]

CAT_FEATURES = ['3', '23', '25']
FEATURES_TO_IMPUT = ['2']
USEFUL_FOR_IMPUT = ['6', '8', '31', '32', '33', '34', '35']


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_lightgbm_pipe(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path, index_col=0)

    X_train = train_data.drop(columns=src.constants.TARGET)
    # '2. Возраст' -> '2'
    X_train.columns = [col_name.split('.')[0] for col_name in X_train.columns]
    for cat_feature in CAT_FEATURES:
        X_train[cat_feature] = X_train[cat_feature].astype('category')
    y_train = train_data[src.constants.TARGET]

    model = get_model()
    model.fit(X_train, y_train)

    joblib.dump(model, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:

    preprocessing = ColumnTransformer(
        transformers=[
            ('knn_imputer', KNNImputer(), FEATURES_TO_IMPUT + USEFUL_FOR_IMPUT),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    lgbm_model = lightgbm.LGBMClassifier(
        # Core Parameters
        objective='multiclass',
        boosting_type='gbdt',
        n_estimators=25,
        learning_rate=.05,
        num_leaves=16,
        n_jobs=-1,
        random_state=src.constants.RANDOM_STATE,

        # Learning Control Parameters
        max_depth=2,
        min_split_gain=.17279224825247272,
        min_child_samples=67,
        verbose=-1,

        categorical_features=CAT_FEATURES,
    )

    pipe = Pipeline(
        [
            ('preprocessing', preprocessing),
            ('lgbm', lgbm_model),
        ]
    )

    return pipe


if __name__ == '__main__':
    train_lightgbm_pipe()
