import pandas as pd
from typing import Dict, Optional, Any
import logging
import numpy as np

class QuestionnairePreprocessor:
    # Default parameters
    DEFAULT_OUTPUT_PID_COL_NAME = 'participant_id'
    DEFAULT_OUTPUT_TRIAL_ID_COL_NAME = 'trial_id'
    DEFAULT_OUTPUT_ITEM_ID_COL_NAME = 'item_id'
    DEFAULT_OUTPUT_RESPONSE_VALUE_COL_NAME = 'response_value'

    # Explicit names for expected configuration keys
    CONFIG_KEY_PID_ORIGINAL = 'participant_id_column_original' # For wide format
    CONFIG_KEY_ITEM_MAP = 'item_column_map'
    CONFIG_KEY_OUTPUT_PID_NAME = 'output_participant_id_col_name'

    # New config keys for handling different input formats
    CONFIG_KEY_INPUT_FORMAT = 'input_format' # 'wide' or 'long'
    CONFIG_KEY_INPUT_PID_COL = 'input_participant_id_col' # For long format
    CONFIG_KEY_INPUT_ITEM_ID_COL = 'input_item_id_col' # For long format
    CONFIG_KEY_INPUT_RESPONSE_VALUE_COL = 'input_response_value_col' # For long format

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("QuestionnairePreprocessor initialized.")

    def extract_items(self,
                      input_df: pd.DataFrame,
                      config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Universally extracts and standardizes questionnaire item data from an input DataFrame.
        Supports both wide and long formats. All column names and mappings are set via config.
        Output is a long-format DataFrame: participant_id, trial_id (optional), item_id, response_value.

        Config keys:
            - input_format: 'wide' or 'long'
            - participant_id_column: str
            - trial_id_column: str (optional, for long format)
            - item_id_column: str (for long format)
            - response_value_column: str (for long format)
            - item_column_map: Dict[str, str] (for wide format: original col -> standard item_id)
            - output_participant_id_col_name: str (optional)
            - output_trial_id_col_name: str (optional)
            - output_item_id_col_name: str (optional)
            - output_response_value_col_name: str (optional)
        """
        self.logger.info(f"QuestionnairePreprocessor - Starting universal item extraction from DataFrame.")
        if input_df is None or input_df.empty:
            self.logger.warning("QuestionnairePreprocessor - Input DataFrame is None or empty. Cannot extract items.")
            return input_df

        input_format = config.get('input_format', 'wide').lower()
        pid_col = config.get('participant_id_column', self.DEFAULT_OUTPUT_PID_COL_NAME)
        trial_col = config.get('trial_id_column', None)
        item_id_col = config.get('item_id_column', None)
        response_col = config.get('response_value_column', None)
        item_map = config.get('item_column_map', {})
        out_pid = config.get('output_participant_id_col_name', self.DEFAULT_OUTPUT_PID_COL_NAME)
        out_trial = config.get('output_trial_id_col_name', self.DEFAULT_OUTPUT_TRIAL_ID_COL_NAME)
        out_item = config.get('output_item_id_col_name', self.DEFAULT_OUTPUT_ITEM_ID_COL_NAME)
        out_resp = config.get('output_response_value_col_name', self.DEFAULT_OUTPUT_RESPONSE_VALUE_COL_NAME)

        if input_format == 'long':
            # Expect columns: pid_col, trial_col (optional), item_id_col, response_col
            required_cols = [pid_col, item_id_col, response_col]
            missing = [col for col in required_cols if col not in input_df.columns]
            if missing:
                self.logger.error(f"QuestionnairePreprocessor - Missing required columns for long format: {missing}")
                return None
            df = input_df.copy()
            # Optionally keep trial_col if present
            cols = [pid_col]
            if trial_col and trial_col in df.columns:
                cols.append(trial_col)
            cols += [item_id_col, response_col]
            df = df[cols].copy()
            # Rename columns to standard names
            rename_map = {str(pid_col): str(out_pid), str(item_id_col): str(out_item), str(response_col): str(out_resp)}
            if trial_col and trial_col in df.columns:
                rename_map[str(trial_col)] = str(out_trial)
            df.rename(columns=rename_map, inplace=True)  # type: ignore[arg-type]
            # Ensure numeric response
            df[out_resp] = pd.to_numeric(df[out_resp], errors='coerce')
            self.logger.info(f"QuestionnairePreprocessor - Extracted long-format questionnaire items. Shape: {df.shape}")
            return pd.DataFrame(df.copy())  # Ensure DataFrame is returned
        else:  # wide format
            # Expect: pid_col, item_map (original col -> standard item_id)
            if pid_col not in input_df.columns:
                self.logger.error(f"QuestionnairePreprocessor - Participant ID column '{pid_col}' not found in the input DataFrame.")
                return None
            if not item_map:
                self.logger.error(f"QuestionnairePreprocessor - 'item_column_map' must be provided for wide format.")
                return None
            # Select and rename columns
            select_rename = {str(pid_col): str(out_pid)}
            for orig_col, std_item in item_map.items():
                if orig_col in input_df.columns:
                    select_rename[str(orig_col)] = str(std_item)
                else:
                    self.logger.warning(f"QuestionnairePreprocessor - Item column '{orig_col}' not found in input. Will be NaN.")
            df = input_df[list(select_rename.keys())].copy()
            df.rename(columns=select_rename, inplace=True)  # type: ignore[arg-type]
            # Melt to long format
            long_df = df.melt(id_vars=[out_pid], var_name=out_item, value_name=out_resp)
            # Remove rows where item_id is the participant_id column
            long_df = long_df[long_df[out_item] != out_pid]
            # Ensure numeric response
            long_df[out_resp] = pd.to_numeric(long_df[out_resp], errors='coerce')
            self.logger.info(f"QuestionnairePreprocessor - Extracted wide-format questionnaire items and converted to long format. Shape: {long_df.shape}")
            return pd.DataFrame(long_df.reset_index(drop=True).copy())  # Ensure DataFrame is returned

    def map_response_values(self, df: pd.DataFrame, value_map: dict, item_col: str = 'item_id', resp_col: str = 'response_value', mapped_col: str = 'response_label') -> pd.DataFrame:
        """
        Map numeric response values to labels/words for each item, as specified in value_map.
        - df: long-format DataFrame (participant_id, trial_id, item_id, response_value)
        - value_map: {item_id: {number: label, ...}, ...}
        - item_col: column with item/question ID
        - resp_col: column with numeric response value
        - mapped_col: new column for mapped label/word
        If no mapping is found for an item, keeps the original value (e.g., for SAM).
        Returns a DataFrame with the new mapped_col.
        """
        def map_func(row):
            item = row[item_col]
            val = row[resp_col]
            mapping = value_map.get(item, None)
            if mapping is not None and val in mapping:
                return mapping[val]
            return val
        df[mapped_col] = df.apply(map_func, axis=1)
        return df