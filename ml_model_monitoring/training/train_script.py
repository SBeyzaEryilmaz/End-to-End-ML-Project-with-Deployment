import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ml_model_monitoring.dataset_ops import DataProcessing
from ml_model_monitoring.utils import get_logger, validate_params


class ModelTraining:
    def __init__(self, model_output_folder) -> None:
        """Constructor for ModelTraining class.

        Args:
            model_output_folder: Folder to save the model.

        Returns:
            None
        """
        self.logger = get_logger(__name__)
        self.preprocessor = DataProcessing()

        self.model_output_folder = model_output_folder

        if not os.path.exists(self.model_output_folder):
            self.logger.info(
                "\tCreating the model output " f"folder: {self.model_output_folder}"
            )
            os.makedirs(self.model_output_folder)

    @validate_params(
        {"dataframe": pd.DataFrame, "cv": int, "test_size": float, "evaluate": bool}
    )
    def train_model(
        self,
        dataframe: pd.DataFrame,
        cv: int,
        test_size: float,
        evaluate: bool = True,
    ) -> Pipeline:
        """Trains the model with specific parameters.

        Args:
            dataframe: Dataframe to the dataset
            test_size: Determines how much of the dataset is allocated as test dataset.
            cv: Number of folds.
            evaluate: Whether to evaluate the model performance.

        Returns:
            Pipeline object, trained model with certain parameters.
        """

        # Preprocessing pipeline
        preprocessing_pipeline = self.preprocessor.create_preprocessing_pipeline(
            dataframe
        )

        self.logger.info("Splitting features and labels")
        X = dataframe.drop("OUTCOME", axis=1)
        y = dataframe["OUTCOME"]

        self.logger.info("Splitting the dataset into train and test sets")
        self.logger.info(f"\tTest size: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.logger.info("Creating the model pipeline")
        gbc_model_pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing_pipeline),
                (
                    "model",
                    GradientBoostingClassifier(
                        learning_rate=0.01,
                        n_estimators=193,
                        max_depth=7,
                        min_samples_split=0.13,
                        min_samples_leaf=0.35,
                    ),
                ),
            ]
        )

        # Cross Validation
        if cv:
            self.logger.info(f"\tPerforming {cv}-fold cross validation")
            scores = cross_val_score(
                gbc_model_pipeline, X=X_train, y=y_train, cv=cv
            ).mean()

            self.logger.info("\t\tSaving the cross validation results")
            # Save the cross validation results
            joblib.dump(
                scores, os.path.join(self.model_output_folder, "cv_scores.joblib")
            )
        else:
            self.logger.info("\tNo cross validation is performed")

        self.logger.info("Fitting the model")
        gbc_model_pipeline.fit(X_train, y_train)

        # Save the model
        joblib.dump(
            gbc_model_pipeline, os.path.join(self.model_output_folder, "model.joblib")
        )

        if evaluate:
            self.logger.info("Evaluating the model")
            self.evaluate_model(X_test, y_test, gbc_model_pipeline)

        self.logger.info("Model training is completed")
        return gbc_model_pipeline

    @validate_params(
        {
            "test_features": pd.DataFrame,
            "test_labels": pd.Series,
            "gbc_model_pipeline": Pipeline,
        }
    )
    def evaluate_model(
        self,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
        gbc_model_pipeline: Pipeline,
    ) -> None:
        """Evaluates the model performance.

        Args:
            test_features: Test features, after the data processing steps
            test_labels: Test labels.
            gbc_model_pipeline: The pipeline where we perform the data processing steps
            and train the model

        Returns:
            None
        """
        self.logger.info("Evaluating the model")
        predictions = gbc_model_pipeline.predict(test_features)

        # Evaluate the performance of the model
        self.logger.info("Saving the classification report")
        report_dict = classification_report(test_labels, predictions, output_dict=True)

        pd.DataFrame(report_dict).to_csv(
            os.path.join(self.model_output_folder, "classification_report.csv")
        )
