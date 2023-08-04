import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

import src


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_data_path', type=click.Path())
def main(input_data_path: str, output_data_path: str) -> None:

    data = pd.read_csv(input_data_path)

    train_data, test_data = train_test_split(
        data, stratify=data[src.constants.TARGET], random_state=src.constants.RANDOM_STATE)

    train_data.to_csv(os.path.join(output_data_path, 'train_data.csv'), index=False)
    train_data.to_csv(os.path.join(output_data_path, 'test_data.csv'), index=False)


if __name__ == '__main__':
    main()
