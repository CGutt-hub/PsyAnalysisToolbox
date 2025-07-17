import pandas as pd
from typing import Optional, Dict, Any, Union, List
import os

# Module-level defaults for TXTReader
# TXT_READER_DEFAULT_FILE_TYPE = "csv" # Moved into class
# TXT_READER_DEFAULT_SHEET_NAME: Union[str, int] = 0 # Moved into class

class TXTReader:
    DEFAULT_FILE_TYPE = "csv"
    DEFAULT_SHEET_NAME: Union[str, int] = 0
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("TXTReader initialized.")

    def load_data(self,
                  file_path: str,
                  reader_type: str = 'tabular', # 'tabular' for CSV/Excel, 'eprime' for E-Prime logs
                  file_type: Optional[str] = None,
                  sheet_name: Optional[Union[str, int]] = None, # For Excel files
                  participant_id_col: Optional[str] = None,
                  **kwargs: Any # For other pandas read_* arguments, e.g., delimiter, header
                  ) -> Optional[pd.DataFrame]:
        """
        Loads tabular data (e.g., questionnaire responses) from a specified file (CSV or Excel).
        Can also parse E-Prime log files if reader_type is 'eprime'.

        Args:
            file_path (str): The full path to the questionnaire data file.
            reader_type (str): The type of reader logic to use. Can be 'tabular' (default)
                               for standard CSV/Excel-like files, or 'eprime' for
                               structured E-Prime log files.
            file_type (Optional[str]): The type of the file ('csv' or 'excel').
                                       If None, defaults to TXTReader.DEFAULT_FILE_TYPE.
                                       For .txt files with a clear delimiter, use 'csv' and specify delimiter in kwargs.
            sheet_name (Optional[Union[str, int]]): Name or index of the sheet to read from (for Excel files).
                                                    If None, defaults to TXTReader.DEFAULT_SHEET_NAME.
            participant_id_col (Optional[str]): If provided and valid, sets this column as the DataFrame index.
                                                If None or invalid, no index is set by this argument.
            **kwargs: Additional keyword arguments to pass to pandas read_csv or read_excel.
                       (e.g., delimiter, header, skiprows).

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the questionnaire data,
                                    or None if loading fails.
        """
        if reader_type.lower() == 'eprime':
            # Use the specialized E-Prime parser
            return self._parse_eprime_log(file_path)

        if not os.path.exists(file_path):
            self.logger.error(f"TXTReader - File not found: {file_path}")
            return None

        # Determine final file_type
        final_file_type = self.DEFAULT_FILE_TYPE
        if file_type is not None:
            file_type_lower = file_type.strip().lower()
            # Treat 'txt' as an alias for 'csv' since we are reading a delimited text file.
            if file_type_lower == 'txt':
                file_type_lower = 'csv'
            if file_type_lower in ['csv', 'excel']:
                final_file_type = file_type_lower
            else:
                self.logger.warning(
                    f"TXTReader: Invalid value ('{file_type}') provided for 'file_type'. "
                    f"Supported types are 'csv', 'txt', or 'excel'. Using default: '{self.DEFAULT_FILE_TYPE}'."
                )

        # Determine final sheet_name (only relevant for excel)
        final_sheet_name = self.DEFAULT_SHEET_NAME
        if final_file_type == 'excel':
            if sheet_name is not None: # User provided a sheet_name
                if isinstance(sheet_name, int):
                    final_sheet_name = sheet_name
                elif isinstance(sheet_name, str) and sheet_name.strip():
                    final_sheet_name = sheet_name.strip()
                else:
                    self.logger.warning(
                        f"TXTReader: Invalid value ('{sheet_name}') provided for 'sheet_name'. "
                        f"Expected an integer or a non-empty string. Using default: '{self.DEFAULT_SHEET_NAME}'."
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

    def _parse_eprime_log(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Internal method to parse E-Prime .txt log files.
        Returns a long-format DataFrame.
        """
        self.logger.info(f"TXTReader - Parsing as E-Prime log file: {file_path}")
        
        header_data: Dict[str, str] = {}
        parsed_responses: List[Dict[str, Any]] = []
        
        current_log_frame_lines: List[str] = []
        in_header = False
        in_log_frame = False
        
        participant_id = None

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_stripped = line.strip()

                    if line_stripped == '*** Header Start ***':
                        in_header = True
                        continue
                    elif line_stripped == '*** Header End ***':
                        in_header = False
                        # More robustly check for participant ID
                        participant_id_raw = header_data.get('Subject')
                        participant_id = participant_id_raw.strip() if isinstance(participant_id_raw, str) else None
                        if not participant_id:
                            self.logger.error(f"Participant ID (Subject) not found or is empty in header of {file_path}. Cannot process questionnaires.")
                            return None
                        continue
                    elif line_stripped.startswith('*** LogFrame Start ***'):
                        in_log_frame = True
                        current_log_frame_lines = []
                        continue
                    elif line_stripped.startswith('*** LogFrame End ***'):
                        in_log_frame = False
                        if current_log_frame_lines:
                            self._process_log_frame(current_log_frame_lines, str(participant_id), parsed_responses)
                        current_log_frame_lines = []
                        continue

                    if in_header:
                        if ':' in line_stripped:
                            key, value = line_stripped.split(':', 1)
                            header_data[key.strip()] = value.strip()
                    elif in_log_frame:
                        current_log_frame_lines.append(line_stripped)
            
            if not parsed_responses:
                self.logger.warning(f"No questionnaire responses extracted from {file_path}.")
                return pd.DataFrame()

            df = pd.DataFrame(parsed_responses)
            self.logger.info(f"Successfully extracted {len(df)} questionnaire responses from {file_path}.")
            return df
        except Exception as e:
            self.logger.error(f"TXTReader - Error parsing E-Prime log file {file_path}: {e}", exc_info=True)
            return None

    def _process_log_frame(self, lines: List[str], participant_id: str, parsed_responses: List[Dict[str, Any]]):
        """
        Processes a single LogFrame's lines to extract questionnaire data.
        """
        log_frame_dict: Dict[str, str] = {k.strip(): v.strip() for k, v in (line.split(':', 1) for line in lines if ':' in line)}
        
        procedure = log_frame_dict.get('Procedure')
        
        # Mapping of procedure names to their respective item/response keys
        proc_map = {
            'panasProc': ('panasList', 'panas', 'panas.RESP', 'PANAS'),
            'bisBasProc': ('bisBasList', 'bis', 'bisBas.RESP', 'BISBAS'),
            'ea11Proc': ('ea11List', 'adjective', 'ea11.RESP', 'EA11'),
            'be7Proc': ('be7List', 'emotion', 'be7.RESP', 'BE7')
        }

        if procedure in proc_map:
            item_id_key, item_text_key, response_key, q_type = proc_map[procedure]
            item_id = log_frame_dict.get(item_id_key)
            item_text = log_frame_dict.get(item_text_key)
            response_value = log_frame_dict.get(response_key)
            if item_id and response_value:
                try:
                    parsed_responses.append({
                        'participant_id': participant_id,
                        'questionnaire_type': q_type,
                        'item_id': item_id,
                        'item_text': item_text,
                        'response_value': int(response_value)
                    })
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert response '{response_value}' to int for item '{item_id}' in {procedure}.")

        elif procedure == 'samProc':
            sam_img = log_frame_dict.get('samBackgroundImg')
            response_value = log_frame_dict.get('SAM.Choice1.Value')
            if sam_img and response_value:
                item_id = None
                if 'samArousal.png' in sam_img: item_id = 'SAM_Arousal'
                elif 'samValence.png' in sam_img: item_id = 'SAM_Valence'
                
                if item_id:
                    try:
                        parsed_responses.append({
                            'participant_id': participant_id, 'questionnaire_type': 'SAM',
                            'item_id': item_id, 'item_text': f'{item_id} Rating',
                            'response_value': int(response_value)
                        })
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert SAM response '{response_value}' to int for item '{item_id}'.")