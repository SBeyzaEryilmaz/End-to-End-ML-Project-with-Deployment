from pathlib import Path

import pandas as pd

from ml_model_monitoring.utils import get_logger, validate_params


class DatasetIO:
    def __init__(self, data_path: Path) -> None:
        """Constructor for DatasetIO class.

        Args:
            data_path: Path to the dataset
        """

        self.logger = get_logger(__name__)

        if not isinstance(data_path, Path):
            error_msg = (
                f"Expected data_path to be a pathlib.Path object, "
                f"but got {type(data_path)}"
            )
            self.logger.error(error_msg)
            raise TypeError(error_msg)

        self.data_path = data_path

    @validate_params({"seperator": str, "encoding": str})
    def safe_read_csv(
        self, seperator: str = ",", encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """Read the dataset from the given path.

        This method assumes that the dataset is separated by commas.

        Returns:
            Pandas DataFrame containing the dataset
        """
        self.logger.info(f"Trying to read the dataset from {self.data_path}")
        self.__validate_path()

        data = pd.read_csv(self.data_path, sep=seperator, encoding=encoding)
        self.logger.info(f"Successfully read the dataset from {self.data_path}")
        return data

    def __validate_path(self) -> None:
        """Validates the given path.

        Returns:
            None

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the path is not a file
            ValueError: If the file is empty
        """

        validation_checks = [
            {
                "condition": lambda: not self.data_path.exists(),
                "error_type": FileNotFoundError,
                "message": f"The file {self.data_path} does not exist.",
            },
            {
                "condition": lambda: not self.data_path.is_file(),
                "error_type": ValueError,
                "message": f"The path {self.data_path} is not a file.",
            },
            {
                "condition": lambda: self.data_path.suffix != ".csv",
                "error_type": ValueError,
                "message": f"The file {self.data_path} is not a CSV file.",
            },
            {
                "condition": lambda: self.data_path.stat().st_size == 0,
                "error_type": ValueError,
                "message": f"The file {self.data_path} is empty.",
            },
        ]

        for check in validation_checks:
            if check["condition"]():  # type: ignore
                self.logger.error(check["message"])
                raise check["error_type"](check["message"])  # type: ignore
