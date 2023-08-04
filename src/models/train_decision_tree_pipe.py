import click
import joblib
import pandas as pd
import numpy as np

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier

import src


CAT_FEATURES = sorted(list(src.constants.CATEGORICAL_FEATURES))
CATEGORIES = [src.constants.CATEGORICAL_FEATURES[feature] for feature in CAT_FEATURES]
FEATURES_TO_IMPUT = [src.constants.QUESTION_2]
USEFUL_FOR_IMPUT = [
    src.constants.QUESTION_6, src.constants.QUESTION_8, src.constants.QUESTION_31,
    src.constants.QUESTION_32, src.constants.QUESTION_33, src.constants.QUESTION_34,
    src.constants.QUESTION_35,
]


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_decision_tree(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path)

    X_train = train_data.drop(columns=src.constants.TARGET)
    y_train = train_data[src.constants.TARGET]

    model = get_model()
    model.fit(X_train, y_train)
    joblib.dump(model, output_model_path)


def get_model() -> sklearn.pipeline.Pipeline:

    oe = OrdinalEncoder(
        categories=CATEGORIES, handle_unknown='use_encoded_value', unknown_value=np.nan)

    preprocessing = ColumnTransformer(
        transformers=[
            ('ordinal_encoding', oe, CAT_FEATURES),
            ('imputer', KNNImputer(), FEATURES_TO_IMPUT + USEFUL_FOR_IMPUT),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')

    decision_tree = DecisionTreeClassifier(
        criterion='log_loss',
        max_depth=4,
        min_samples_split=151,
        min_samples_leaf=193,
        min_weight_fraction_leaf=0.17600000000000002,
        random_state=src.constants.RANDOM_STATE,
        max_leaf_nodes=3,
        min_impurity_decrease=0.216,
    )

    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('decision_tree', decision_tree),
    ])

    return pipe


if __name__ == '__main__':
    train_decision_tree()
