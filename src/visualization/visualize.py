import click
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

import src


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('input_pipe_path', type=click.Path(exists=True))
@click.argument('output_figure_path', type=click.Path())
def main(input_data_path: str, input_pipe_path: str, output_figure_path: str) -> None:

    X = pd.read_csv(input_data_path).drop(columns=src.constants.TARGET)

    catboost_pipe = joblib.load(input_pipe_path)
    catboost_model = catboost_pipe.named_steps['catboost_model']
    preprocessing = catboost_pipe.named_steps['preprocessing']

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
    plt.savefig(output_figure_path)


if __name__ == '__main__':
    main()
