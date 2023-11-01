import click
import joblib
import pandas as pd

import src


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_model_path', type=click.Path())
def train_my_decision_train(input_data_path: str, output_model_path: str) -> None:

    train_data = pd.read_csv(input_data_path, index_col=0)

    X_train = train_data.drop(columns=src.constants.TARGET)
    y_train = train_data[src.constants.TARGET]

    model = src.my_decision_tree.MyDecisionTreeClassifier(
        min_samples_leaf=1, min_samples_split=2)
    model.fit(
        X_train, y_train,
        categorical_feature_names=src.constants.CATEGORICAL_FEATURES | src.constants.BINARY_FEATURES,
        rank_feature_names=src.constants.RANK_FEATURES,
        numerical_feature_names=src.constants.NUMERICAL_FEATURES,
    )
    joblib.dump(model, output_model_path)


if __name__ == '__main__':
    train_my_decision_train()
