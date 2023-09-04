import logging
import os

import joblib
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report
from pandas import DataFrame

from ml_model_monitoring.utils import validate_params


class Monitoring:
    def __init__(self, output_folder) -> None:
        """Constructor for ModelTraining class.

        Args:
            output_folder: Folder to save the results

        Returns:
            None

        Raises:
            TypeError: If 'dataset' is not  of a pathlib.Path object
        """

        self.logger = logging.getLogger(__name__)
        self.output_folder = output_folder

        if not isinstance(output_folder, str):
            error_msg = (
                f"Expected output_folder to be a string, "
                f"but got {type(output_folder)}"
            )
            self.logger.error(error_msg)
            raise TypeError(error_msg)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @validate_params({"reference_dataset": DataFrame, "target_dataset": DataFrame})
    def data_drift(self, reference_dataset, target_dataset) -> None:
        """Gives a visual report on the statistical distribution of the data
        over time.

        Args:
            reference_dataset: Reference dataset
            target_dataset: Target dataset

        Returns:
            None
        """
        data_drift_report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        data_drift_report.run(
            reference_data=reference_dataset, current_data=target_dataset
        )
        data_drift_report.save_html(
            os.path.join(self.output_folder, "data_drift_report.html")
        )

    @validate_params(
        {
            "reference_dataset": DataFrame,
            "target_dataset": DataFrame,
            "model_output_path": str,
        }
    )
    def model_monitoring(
        self, reference_dataset, target_dataset, model_output_path
    ) -> None:
        """Visually demonstrates the model performance.

        Args:
            reference_dataset: Reference dataset
            target_dataset: Target dataset
            model_output_path: Path where the model is

        Returns:
            None
        """

        gbc_model_pipeline = joblib.load(model_output_path + "/model.joblib")

        reference_dataset["prediction"] = gbc_model_pipeline.predict(reference_dataset)
        target_dataset["prediction"] = gbc_model_pipeline.predict(target_dataset)

        column_mapping = ColumnMapping()

        column_mapping.target = "OUTCOME"
        column_mapping.prediction = "prediction"

        classification_performance_report = Report(metrics=[ClassificationPreset()])

        classification_performance_report.run(
            reference_data=reference_dataset,
            current_data=target_dataset,
            column_mapping=column_mapping,
        )
        (
            classification_performance_report.save_html(
                os.path.join(
                    self.output_folder, "classification_performance_report.html"
                )
            )
        )
