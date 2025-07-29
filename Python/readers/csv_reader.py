"""
CSV Reader Module
----------------
Reads and parses CSV data files.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class CSVReader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("CSVReader initialized.")

    def load_data(self, file_path: str, config: Dict[str, Any]) -> pd.DataFrame:
        self.logger.info(f"CSVReader: Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path, **config)
            self.logger.info(f"CSVReader: Loaded CSV file with shape {df.shape}.")
            return df
        except Exception as e:
            self.logger.error(f"CSVReader: Failed to load CSV file: {e}", exc_info=True)
            raise