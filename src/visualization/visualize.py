import os

import click
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

import my_ds_tools
import src


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('input_pipe_path', type=click.Path(exists=True))
def visualize(input_data_path: str, input_pipe_path: str) -> None:

    data = pd.read_csv(input_data_path, index_col=0)
    X = data.drop(columns=src.constants.TARGET)
    y = data[src.constants.TARGET]

    catboost_pipe = joblib.load(input_pipe_path)
    catboost_model = catboost_pipe.named_steps['catboost_model']
    preprocessing = catboost_pipe.named_steps['preprocessing']

    # SHAP values
    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(preprocessing.transform(X))

    shap.summary_plot(
        shap_values=shap_values,
        features=X,
        class_names=['доброкачественная опухоль', 'злокачественная опухоль', 'норма'],
        max_display=33,
        plot_size=(16, 10),
        show=False,
    )
    plt.savefig(os.path.join('reports', 'figures', 'SHAP_values.png'))

    # Permutation Importance
    pi = permutation_importance(
        estimator=catboost_pipe,
        X=X,
        y=y,
        scoring='accuracy',
        n_repeats=1000,
        n_jobs=-1,
        random_state=src.constants.RANDOM_STATE,
    )
    fig, ax = my_ds_tools.interpretation.permutation_importance_plot(pi, X.columns)
    fig.savefig(
        os.path.join('reports', 'figures', 'Permutation_Importance.png'),
        bbox_inches='tight',
    )


if __name__ == '__main__':
    visualize()
