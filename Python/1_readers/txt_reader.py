import pandas as pd
from typing import Optional, Dict, Any, Union
import os

# Module-level defaults for TXTReader
TXT_READER_DEFAULT_FILE_TYPE = "csv"
TXT_READER_DEFAULT_SHEET_NAME: Union[str, int] = 0

class TXTReader:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("TXTReader initialized.")

    def load_data(self,
                  file_path: str,
                  file_type: Optional[str] = None,
                  sheet_name: Optional[Union[str, int]] = None, # For Excel files
                  participant_id_col: Optional[str] = None,
                  **kwargs: Any # For other pandas read_* arguments, e.g., delimiter, header
                  ) -> Optional[pd.DataFrame]:
        """
        Loads tabular data (e.g., questionnaire responses) from a specified file (CSV or Excel).
        While named for questionnaires, this loader is suitable for any data in these
        supported tabular formats. For .txt files that are structured like CSVs (e.g., delimiter-separated),
        use file_type='csv' and pass the appropriate delimiter via **kwargs (e.g., delimiter='\\t').

        Args:
            file_path (str): The full path to the questionnaire data file.
            file_type (Optional[str]): The type of the file ('csv' or 'excel').
                                       If None, defaults to TXT_READER_DEFAULT_FILE_TYPE.
                                       For .txt files with a clear delimiter, use 'csv' and specify delimiter in kwargs.
            sheet_name (Optional[Union[str, int]]): Name or index of the sheet to read from (for Excel files).
                                                    If None, defaults to TXT_READER_DEFAULT_SHEET_NAME.
            participant_id_col (Optional[str]): If provided and valid, sets this column as the DataFrame index.
                                                If None or invalid, no index is set by this argument.
            **kwargs: Additional keyword arguments to pass to pandas read_csv or read_excel.
                       (e.g., delimiter, header, skiprows).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the questionnaire data,
                                    or None if loading fails.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"TXTReader - File not found: {file_path}")
            return None

        # Determine final file_type
        final_file_type = TXT_READER_DEFAULT_FILE_TYPE
        if file_type is not None:
            if isinstance(file_type, str) and file_type.strip().lower() in ['csv', 'excel']:
                final_file_type = file_type.strip().lower()
            else:
                self.logger.warning(
                    f"TXTReader: Invalid value ('{file_type}') provided for 'file_type'. "
                    f"Supported types are 'csv' or 'excel'. Using default: '{TXT_READER_DEFAULT_FILE_TYPE}'."
                )

        # Determine final sheet_name (only relevant for excel)
        final_sheet_name = TXT_READER_DEFAULT_SHEET_NAME
        if final_file_type == 'excel':
            if sheet_name is not None: # User provided a sheet_name
                if isinstance(sheet_name, int):
                    final_sheet_name = sheet_name
                elif isinstance(sheet_name, str) and sheet_name.strip():
                    final_sheet_name = sheet_name.strip()
                else:
                    self.logger.warning(
                        f"TXTReader: Invalid value ('{sheet_name}') provided for 'sheet_name'. "
                        f"Expected an integer or a non-empty string. Using default: '{TXT_READER_DEFAULT_SHEET_NAME}'."
                    )
        
        # Determine final participant_id_col
        final_pid_col: Optional[str] = None # Default behavior is no specific PID column to set as index initially
        if participant_id_col is not None: # User provided a participant_id_col
            if isinstance(participant_id_col, str) and participant_id_col.strip():
                final_pid_col = participant_id_col.strip()
            else:
                self.logger.warning(
                    f"TXTReader: Invalid value ('{participant_id_col}') provided for 'participant_id_col'. "
                    f"Expected a non-empty string. Will not attempt to set index by this column."
                )

        load_info = f"file: {file_path}, type: {final_file_type}"
        if final_file_type == 'excel':
            load_info += f", sheet: {final_sheet_name}"
        self.logger.info(f"TXTReader - Attempting to load data from: {load_info}")

        try:
            df: Optional[pd.DataFrame] = None
            if final_file_type == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif final_file_type == 'excel':
                df = pd.read_excel(file_path, sheet_name=final_sheet_name, **kwargs)
            # No else needed here as final_file_type is guaranteed to be 'csv' or 'excel'

            if df is not None:
                self.logger.info(f"TXTReader - Successfully loaded data with shape: {df.shape}")
                if final_pid_col:
                    if final_pid_col in df.columns:
                        df = df.set_index(final_pid_col)
                        self.logger.info(f"TXTReader - Set '{final_pid_col}' as index.")
                    else:
                        self.logger.warning(f"TXTReader - Participant ID column '{final_pid_col}' not found in the loaded data. Index not set.")
            return df

        except FileNotFoundError:
            self.logger.error(f"TXTReader - File not found during pandas read: {file_path}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"TXTReader - Error loading data from {file_path}: {e}", exc_info=True)
            return None