import click
import pandas as pd

import src


@click.command()
@click.argument('etled_data_path', type=click.Path(exists=True))
@click.argument('encoded_data_path', type=click.Path())
def encoding(etled_data_path: str, encoded_data_path: str) -> None:

    data = pd.read_csv(etled_data_path, index_col=0)

    # кодируем бинарные признаки {YES: 1, NO: 0}
    for bin_feature in src.constants.BINARY_FEATURES:
        data[bin_feature].replace(
            {src.constants.YES: 1, src.constants.NO: 0}, inplace=True)

    # кодируем ранговые признаки
    for rank_feature in src.constants.RANK_FEATURES:
        mapping_dict = {
            value: i
            for i, value in enumerate(src.constants.RANK_FEATURES[rank_feature])
        }
        data[rank_feature].replace(mapping_dict, inplace=True)

    # # кодируем категориальные признаки
    # for cat_feature, categories in src.constants.CATEGORICAL_FEATURES.items():
    #     data[cat_feature] = pd.Categorical(data[cat_feature], categories=categories)
    #     # категории кодируются числами 0, 1, 2, ... np.NaN числом -1
    #     data[cat_feature] = data[cat_feature].cat.codes

    data.to_csv(encoded_data_path)


if __name__ == '__main__':
    encoding()
