"""
TXT Reader Module
----------------
Reads and parses TXT questionnaire or data files.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class TXTReader:
    """
    Reads and parses TXT questionnaire or data files.
    - Accepts config dict for reading parameters.
    - Returns parsed data as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("TXTReader initialized.")

    def load_data(self, file_path: str, reader_type: str, file_type: str, delimiter: str, encoding: str) -> pd.DataFrame:
        """
        Loads and parses a TXT file using the provided parameters.
        Returns a DataFrame with the parsed data.
        Tries multiple strategies for robustness to inconsistent columns.
        """
        self.logger.info(f"TXTReader: Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            self.logger.info(f"TXTReader: Loaded TXT file with shape {df.shape} (read_csv).")
            return df
        except Exception as e:
            self.logger.warning(f"TXTReader: read_csv failed: {e}. Trying read_fwf.")
            try:
                df = pd.read_fwf(file_path, encoding=encoding)
                self.logger.info(f"TXTReader: Loaded TXT file with shape {df.shape} (read_fwf).")
                return df
            except Exception as e2:
                self.logger.warning(f"TXTReader: read_fwf failed: {e2}. Trying fallback line-by-line parsing.")
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    # Try to split by delimiter, fallback to whitespace
                    rows = [line.strip().split(delimiter) if delimiter else line.strip().split() for line in lines if line.strip()]
                    max_cols = max(len(row) for row in rows)
                    # Pad rows to max_cols
                    rows = [row + [''] * (max_cols - len(row)) for row in rows]
                    df = pd.DataFrame(rows)
                    self.logger.info(f"TXTReader: Loaded TXT file with shape {df.shape} (fallback line-by-line parsing).")
                    return df
                except Exception as e3:
                    self.logger.error(f"TXTReader: All loading methods failed: {e3}", exc_info=True)
                    raise