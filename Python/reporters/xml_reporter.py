"""
XML Reporter Module
------------------
Handles saving DataFrames to XML files for reporting.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Any

class XMLReporter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("XMLReporter initialized.")

    def save_dataframe(self, data_df: pd.DataFrame, output_dir: str, filename: str) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        data_df.to_xml(path, index=False)
        self.logger.info(f"XMLReporter: Saved DataFrame to {path}.")