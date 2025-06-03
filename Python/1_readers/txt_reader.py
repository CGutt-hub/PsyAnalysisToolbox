import pandas as pd
from typing import Optional, Dict, Any
import os

class TXTReader:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("TXTReader initialized.")

    def load_data(self,
                  file_path: str,
                  file_type: str = 'csv',
                  sheet_name: Optional[str | int] = 0, # For Excel files
                  participant_id_col: Optional[str] = None, # Optional: column name for participant ID
                  **kwargs: Any # For other pandas read_* arguments
                  ) -> Optional[pd.DataFrame]:
        """
        Loads tabular data (e.g., questionnaire responses) from a specified file (CSV or Excel).
        While named for questionnaires, this loader is suitable for any data in these
        supported tabular formats. For .txt files that are structured like CSVs (e.g., delimiter-separated),
        use file_type='csv' and pass the appropriate delimiter via **kwargs (e.g., delimiter='\\t').

        Args:
            file_path (str): The full path to the questionnaire data file.
            file_type (str): The type of the file ('csv' or 'excel'). Defaults to 'csv'.
                              For .txt files with a clear delimiter, use 'csv' and specify the delimiter in kwargs.
            sheet_name (Optional[str | int]): Name or index of the sheet to read from (for Excel files).
                                               Defaults to 0 (the first sheet).
            participant_id_col (Optional[str]): If provided, sets this column as the DataFrame index.
            **kwargs: Additional keyword arguments to pass to pandas read_csv or read_excel.
                       (e.g., delimiter, header, skiprows).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the questionnaire data,
                                    or None if loading fails.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"TXTReader - File not found: {file_path}")
            return None

        self.logger.info(f"TXTReader - Attempting to load data from: {file_path} (type: {file_type})")
        try:
            df: Optional[pd.DataFrame] = None
            if file_type.lower() == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_type.lower() == 'excel':
                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            else:
                self.logger.error(f"TXTReader - Unsupported file type: {file_type}. Supported types are 'csv' and 'excel'.")
                return None

            if df is not None:
                self.logger.info(f"TXTReader - Successfully loaded data with shape: {df.shape}")
                if participant_id_col:
                    if participant_id_col in df.columns:
                        df = df.set_index(participant_id_col)
                        self.logger.info(f"TXTReader - Set '{participant_id_col}' as index.")
                    else:
                        self.logger.warning(f"TXTReader - Participant ID column '{participant_id_col}' not found in the loaded data.")
            return df

        except FileNotFoundError:
            self.logger.error(f"TXTReader - File not found during pandas read: {file_path}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"TXTReader - Error loading data from {file_path}: {e}", exc_info=True)
            return None