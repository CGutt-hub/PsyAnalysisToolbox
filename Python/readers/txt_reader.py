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
        if not os.path.exists(file_path):
            self.logger.error(f"TXTReader - File not found: {file_path}")
            return None

        # Get encoding from kwargs to pass to the correct parser.
        encoding = kwargs.get('encoding', 'utf-8')

        # --- Main logic based on reader_type ---
        if reader_type.lower() == 'eprime':
            return self._parse_eprime_log(file_path, encoding=encoding)

        elif reader_type.lower() == 'tabular':
            # Determine final file_type for tabular reading
            final_file_type = self.DEFAULT_FILE_TYPE
            if file_type is not None:
                file_type_lower = file_type.strip().lower()
                if file_type_lower == 'txt':
                    file_type_lower = 'csv' # Treat 'txt' as an alias for 'csv'
                if file_type_lower in ['csv', 'excel']:
                    final_file_type = file_type_lower
                else:
                    self.logger.warning(
                        f"TXTReader: Invalid 'file_type' ('{file_type}'). Using default: '{self.DEFAULT_FILE_TYPE}'."
                    )

            # Determine final sheet_name (only relevant for excel)
            final_sheet_name = sheet_name if sheet_name is not None else self.DEFAULT_SHEET_NAME

            load_info = f"file: {file_path}, type: {final_file_type}"
            if final_file_type == 'excel':
                load_info += f", sheet: {final_sheet_name}"
            self.logger.info(f"TXTReader - Attempting to load tabular data from: {load_info}")

            try:
                df: Optional[pd.DataFrame] = None
                if final_file_type == 'csv':
                    df = pd.read_csv(file_path, **kwargs)
                elif final_file_type == 'excel':
                    df = pd.read_excel(file_path, sheet_name=final_sheet_name, **kwargs)

                if df is not None:
                    self.logger.info(f"TXTReader - Successfully loaded data with shape: {df.shape}")
                    if participant_id_col and participant_id_col in df.columns:
                        df = df.set_index(participant_id_col)
                        self.logger.info(f"TXTReader - Set '{participant_id_col}' as index.")
                return df

            except FileNotFoundError:
                self.logger.error(f"TXTReader - File not found during pandas read: {file_path}", exc_info=True)
                return None
            except Exception as e:
                self.logger.error(f"TXTReader - Error loading tabular data from {file_path}: {e}", exc_info=True)
                return None
        else:
            self.logger.error(f"TXTReader - Unsupported reader_type: '{reader_type}'. Must be 'eprime' or 'tabular'.")
            return None

    def _parse_eprime_log(self, file_path: str, encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
        """
        Internal method to parse E-Prime .txt log files.
        Args:
            file_path (str): The full path to the E-Prime log file.
            encoding (str): The character encoding of the file (e.g., 'utf-8', 'utf-16').
        Returns a long-format DataFrame.
        """
        self.logger.info(f"TXTReader - Parsing as E-Prime log file: {file_path} with encoding '{encoding}'")
        
        header_data: Dict[str, str] = {}
        parsed_responses: List[Dict[str, Any]] = []
        
        current_log_frame_lines: List[str] = []
        in_header = False
        in_log_frame = False
        last_condition = "baseline" # Default context for pre-experiment questionnaires
        condition_counters = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Training': 0}
        
        participant_id = None

        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                for line in f:
                    line_stripped = line.strip()

                    if line_stripped == '*** Header Start ***':
                        in_header = True
                        continue
                    elif line_stripped == '*** Header End ***':
                        in_header = False
                        # More robustly check for participant ID
                        participant_id_raw = header_data.get('subject')
                        participant_id = participant_id_raw.strip() if isinstance(participant_id_raw, str) else None
                        if not participant_id:
                            self.logger.error(f"Participant ID (Subject) not found or is empty in header of {file_path}. Cannot process questionnaires.")
                            return None  # Exit early if we can't get a valid PID
                        continue
                    elif line_stripped.startswith('*** LogFrame Start ***'):
                        in_log_frame = True
                        current_log_frame_lines = []
                        continue
                    elif line_stripped.startswith('*** LogFrame End ***'):
                        in_log_frame = False
                        if current_log_frame_lines:
                            # The processor is now stateful and can update the condition
                            last_condition = self._process_log_frame(current_log_frame_lines, str(participant_id), parsed_responses, last_condition, condition_counters)
                        current_log_frame_lines = []
                        continue

                    if in_header:
                        if ':' in line_stripped:
                            key, value = line_stripped.split(':', 1)
                            header_data[key.strip().lower()] = value.strip()
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

    def _process_log_frame(self, lines: List[str], participant_id: str, parsed_responses: List[Dict[str, Any]], last_condition: str, condition_counters: Dict[str, int]) -> str:
        """
        Processes a single LogFrame's lines to extract questionnaire data. It is
        stateful and returns the updated condition context for the next frame.
        """
        log_frame_dict: Dict[str, str] = {k.strip().lower(): v.strip() for k, v in (line.split(':', 1) for line in lines if ':' in line)}

        # Get the procedure name and convert it to lowercase for case-insensitive matching.
        procedure_raw = log_frame_dict.get('procedure')
        if not procedure_raw:
            # This frame has no 'Procedure' line, so we can't process it.
            return last_condition
        procedure = procedure_raw.lower()

        # --- State Update: Check if this frame defines a new condition ---
        if procedure in ['trialproc', 'movietrialproc', 'movietrainingproc']:
            new_condition = last_condition  # Default to old condition if not found
            # First, check for an explicit 'condition' key
            if 'condition' in log_frame_dict:
                new_condition = log_frame_dict.get('condition', last_condition)
            # If not, check for a movie filename to infer the condition
            elif 'moviefilename' in log_frame_dict:
                filename = log_frame_dict.get('moviefilename', '').upper()
                if filename.startswith('POS'):
                    new_condition = 'Positive'
                elif filename.startswith('NEG'):
                    new_condition = 'Negative'
                elif filename.startswith('NEU'):
                    new_condition = 'Neutral'
                elif filename.startswith('TRAI'):
                    # The training trial is a special case. We can label it 'Training'.
                    new_condition = 'Training'
                else:
                    self.logger.warning(f"Could not determine condition from movie filename: {filename}")
            
            # Increment the counter for the new condition
            if new_condition in condition_counters:
                condition_counters[new_condition] += 1
                self.logger.debug(f"Incremented counter for '{new_condition}' to {condition_counters[new_condition]}.")
            
            if new_condition != last_condition:
                self.logger.debug(f"Parser state: updated last_condition from '{last_condition}' to '{new_condition}'")
            return new_condition # Return the new state for the next frame

        # --- Process SAM ratings using the current state (last_condition) ---
        if procedure == 'samproc':
            sam_img = log_frame_dict.get('sambackgroundimg')
            response_value = log_frame_dict.get('sam.choice1.value')
            if sam_img and response_value:
                item_id_base = None
                if 'samarousal.png' in sam_img.lower(): item_id_base = 'SAM_Arousal'
                elif 'samvalence.png' in sam_img.lower(): item_id_base = 'SAM_Valence'
                if item_id_base:
                    # Exclude training trials from being added to the results, as requested.
                    if last_condition == 'Training':
                        self.logger.debug("Skipping SAM rating for 'Training' trial.")
                        return last_condition # Return state unchanged

                    # Get the current trial number for this condition
                    trial_num = condition_counters.get(last_condition, 1)
                    try:
                        parsed_responses.append({
                            'participant_id': participant_id, 'questionnaire_type': 'SAM',
                            'item_id': f"{item_id_base}_{last_condition}_{trial_num}", 'item_text': f'{item_id_base} Rating ({last_condition} Trial {trial_num})',
                            'response_value': int(response_value)
                        })
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert SAM response '{response_value}' to int for item '{item_id_base}'.")
            return last_condition

        # --- Process standard questionnaires (PANAS, BIS/BAS, STAI, EA11, BE7) ---
        # This map now includes the correct response key and a flag for post-trial context.
        proc_map = {
            # proc_name: (item_number_key, item_text_key, response_key, q_type, item_id_prefix, is_post_trial)
            'panasproc': ('panaslist', 'panas', 'panas.choice1.value', 'PANAS', 'panas', False),
            'bisbasproc': ('bisbaslist', 'bis', 'bisbas.choice1.value', 'BISBAS', 'bis', False),
            'staiproc': ('stailist', 'stai', 'stai.choice1.value', 'STAI', 'stai', False),
            'ea11proc': ('ea11list', 'adjective', 'ea11.choice1.value', 'EA11', 'ea', True),
            'be7proc': ('be7list', 'emotion', 'be7.choice1.value', 'BE7', 'be', True)
        }

        if procedure in proc_map:
            item_num_key, item_text_key, response_key, q_type, item_id_prefix, is_post_trial = proc_map[procedure]
            item_num = log_frame_dict.get(item_num_key)
            response_value = log_frame_dict.get(response_key)
            
            if item_num and response_value:
                base_item_id = f"{item_id_prefix}{item_num}"
                
                if is_post_trial:
                    # For post-trial items, add condition and trial number context.
                    if last_condition == 'Training':
                        self.logger.debug(f"Skipping post-trial item '{base_item_id}' for 'Training' trial.")
                        return last_condition # Return state unchanged
                    
                    trial_num = condition_counters.get(last_condition, 1)
                    final_item_id = f"{base_item_id}_{last_condition}_{trial_num}"
                    item_text = f"{log_frame_dict.get(item_text_key, '')} ({last_condition} Trial {trial_num})"
                else:
                    # For pre-experiment items, use the simple ID.
                    final_item_id = base_item_id
                    item_text = log_frame_dict.get(item_text_key, '')

                try:
                    parsed_responses.append({
                        'participant_id': participant_id, 'questionnaire_type': q_type,
                        'item_id': final_item_id, 'item_text': item_text,
                        'response_value': int(response_value)
                    })
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert response '{response_value}' to int for item '{final_item_id}' in {procedure}.")
        
        # For any other frame type, just return the condition state unchanged.
        return last_condition