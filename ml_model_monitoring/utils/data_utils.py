import os
import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_data(
    data: pd.DataFrame, features: list, output_folder: str, target_column: str
) -> None:
    """Visualize the data.

    Args:
        data: Pandas DataFrame containing the dataset
        features: List of features to visualize
        output_folder: Folder to save the plots
        target_column: Target column of the dataset

    Returns:
        None
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        for feature in features:
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x=feature, hue=target_column, data=data, palette="Set3")
            ax.set_title(f"{feature} Distribution")
            plt.xticks(rotation=45)
            plt.legend(title=target_column)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{feature}_distribution.png"))
            plt.close()


def map_columns(data: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    data[column] = data[column].map(mapping)
    return data


def clean_column(data: pd.DataFrame, column: str, to_replace: list) -> pd.DataFrame:
    for item in to_replace:
        data[column] = data[column].str.replace(item, "")
    return data


def drop_columns(dataframe, columns: list) -> pd.DataFrame:
    return dataframe.drop(columns, axis=1)
