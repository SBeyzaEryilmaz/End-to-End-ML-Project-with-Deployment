import argparse
from pathlib import Path

from ml_model_monitoring.dataset_ops import DataProcessing, DatasetIO
from ml_model_monitoring.monitoring import Monitoring
from ml_model_monitoring.training import ModelTraining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Model Monitoring")
    parser.add_argument(
        "--ref_data_path",
        type=str,
        default="data/reference_dataset.csv",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--tar_data_path",
        type=str,
        default="data/target_dataset.csv",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--model_output_folder",
        type=str,
        default="models",
        help="Folder to save the model",
    )
    parser.add_argument(
        "--monitoring_output_folder",
        type=str,
        default="monitoring_results",
        help="Folder to save the monitoring results",
    )

    parser.add_argument(
        "--data_visualization",
        type=bool,
        default=True,
        help="Whether to visualize the data",
    )
    parser.add_argument(
        "--train_model",
        type=bool,
        default=True,
        help="Whether to train the model",
    )
    parser.add_argument(
        "--monitor_model",
        type=bool,
        default=True,
        help="Whether to monitor the model",
    )
    args = parser.parse_args()

    ref_io_ops = DatasetIO(Path(args.ref_data_path))
    reference_dataset = ref_io_ops.safe_read_csv(seperator=",", encoding="utf-8")

    # This is a copy of the reference dataset, which will be used for monitoring.
    reference_dataset_copy = reference_dataset.copy()

    tar_io_ops = DatasetIO(Path(args.tar_data_path))
    target_dataset = tar_io_ops.safe_read_csv(seperator=",", encoding="utf-8")

    data_processing = DataProcessing()
    if args.data_visualization:
        data_processing.data_visualize(reference_dataset, "visualizations")

    if args.train_model:
        model_training = ModelTraining(args.model_output_folder)
        model_training.train_model(dataframe=reference_dataset, cv=5, test_size=0.2)

    if args.monitor_model:
        target_dataset = data_processing.preprocess_data(target_dataset)
        reference_dataset = data_processing.preprocess_data(reference_dataset_copy)

        monitor = Monitoring(args.monitoring_output_folder)
        monitor.data_drift(reference_dataset, target_dataset)
        monitor.model_monitoring(
            reference_dataset, target_dataset, args.model_output_folder
        )
