import pandas as pd
import os
from typing import Optional, Any


class CSVReader:
    """
    A dedicated reader for loading data from CSV files into pandas DataFrames.
    """

    def __init__(self, logger):
        """
        Initializes the CSVReader.

        Args:
            logger: A logging.Logger instance.
        """
        self.logger = logger
        self.logger.info("CSVReader initialized.")

    def load_data(self, file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """
        Loads data from a specified CSV file.

        Args:
            file_path (str): The full path to the CSV data file.
            **kwargs: Additional keyword arguments to pass to pandas.read_csv
                      (e.g., delimiter, header, encoding, index_col).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                    or None if loading fails.
        """
        if not isinstance(file_path, str) or not file_path.strip():
            self.logger.error("CSVReader: 'file_path' must be a non-empty string.")
            return None

        if not os.path.exists(file_path):
            self.logger.error(f"CSVReader - File not found: {file_path}")
            return None

        self.logger.info(f"CSVReader - Attempting to load data from: {file_path}")

        try:
            df = pd.read_csv(file_path, **kwargs)
            self.logger.info(f"CSVReader - Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"CSVReader - Error loading data from {file_path}: {e}", exc_info=True)
            return None