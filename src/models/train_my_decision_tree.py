import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

import src


def train_my_decision_train(input_data_path: str, output_model_path: str) -> None:
    data = pd.read_csv(input_data_path)
    X = data.drop(columns=src.constants.TARGET)
    y = data[src.constants.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=src.constants.RANDOM_STATE)

    model = src.my_decision_tree.MyDecisionTreeClassifier(
        min_samples_leaf=1, min_samples_split=2)
    model.fit(
        X_train, y_train,
        categorical_feature_names=src.constants.CATEGORICAL_FEATURES|src.constants.BINARY_FEATURES,
        rank_feature_names=src.constants.RANK_FEATURES,
        numerical_feature_names=src.constants.NUMERICAL_FEATURES,
    )
    joblib.dump(model, output_model_path)


if __name__ == '__main__':
    train_my_decision_train(
        src.constants.ENCODED_DATA_PATH, src.constants.MY_DECISION_TREE_MODEL_PATH)
