import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ml_model_monitoring.utils import (
    clean_column,
    get_logger,
    map_columns,
    visualize_data,
)
from ml_model_monitoring.utils.data_utils import drop_columns


class DataProcessing:
    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def data_visualize(self, dataframe, output_folder: str) -> None:
        """Visualize the data.

        Args:
            dataframe: Pandas DataFrame containing the dataset
            output_folder: Folder to save the plots

        Returns:
            None
        """
        self.logger.info("Visualizing the data")
        visualize_data(
            dataframe,
            features=[
                "AGE",
                "GENDER",
                "RACE",
                "DRIVING_EXPERIENCE",
                "EDUCATION",
                "INCOME",
                "VEHICLE_YEAR",
                "VEHICLE_TYPE",
            ],
            output_folder=output_folder,
            target_column="OUTCOME",
        )
        self.logger.info("Data visualization completed")

    def preprocess_data(self, dataframe) -> pd.DataFrame:
        """Preprocess the data.

        Args:
            dataframe: Pandas DataFrame containing the dataset

        Returns:
            Preprocessed Pandas DataFrame
        """
        self.logger.info("Preprocessing the data")

        column_mappings = {
            "AGE": {
                "16-25": "young",
                "26-39": "adult",
                "40-64": "middle-aged",
                "65+": "old",
            },
            "DRIVING_EXPERIENCE": {
                "0-9y": "competent",
                "10-19y": "experienced",
                "20-29y": "professional",
                "30y+": "expert",
            },
        }

        self.logger.info("\tMapping the columns")
        for column, mapping in column_mappings.items():
            self.logger.info(f"\t\tMapping the column: {column}")
            dataframe = map_columns(dataframe, column, mapping)

        self.logger.info("\tCleaning the columns")
        dataframe = clean_column(dataframe, "VEHICLE_YEAR", ["after", "before"])

        self.logger.info("\tDropping the columns")
        dataframe = drop_columns(dataframe, ["ID"])
        return dataframe

    def create_preprocessing_pipeline(self, dataframe) -> ColumnTransformer:
        """Creates a preprocessing pipeline.

        Args:
            dataframe: Pandas DataFrame containing the dataset

        Returns:
            Preprocessing pipeline
        """
        self.logger.info("Creating the preprocessing pipeline")
        dataframe = self.preprocess_data(dataframe)

        numeric_cols = list(dataframe.select_dtypes(include=np.number).columns)
        cat_cols = list(dataframe.select_dtypes(exclude=np.number).columns)

        numeric_cols.remove("OUTCOME")

        self.logger.info("Creating the numerical transformer")
        numeric_transformer = Pipeline(
            [
                ("numeric_imputer", SimpleImputer(strategy="median")),
                ("numeric_scaler", StandardScaler()),
            ]
        )

        self.logger.info("Creating the categorical transformer")
        categorical_transformer = Pipeline(
            [
                ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal_encoding",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        self.logger.info("Mapping the columns with the transformers")
        return ColumnTransformer(
            [
                ("numeric", numeric_transformer, numeric_cols),
                ("categorical", categorical_transformer, cat_cols),
            ]
        )
