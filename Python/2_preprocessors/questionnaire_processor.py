import pandas as pd
from typing import Dict, Optional, Any

class QuestionnairePreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("QuestionnairePreprocessor initialized.")

    def extract_items(self,
                      input_df: pd.DataFrame,
                      config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Extracts and standardizes questionnaire item data from an input DataFrame.

        Args:
            input_df (pd.DataFrame): The DataFrame containing raw questionnaire data
                                     (e.g., as loaded by TXTReader).
            config (Dict[str, Any]): Configuration dictionary. Expected keys:
                'participant_id_column_original' (str): Name of the PID column in the source file.
                'item_column_map' (Dict[str, str]): Maps original item column names to standard names.
                                                    e.g., {"Q1_raw": "item_1", "PANAS_01_resp": "panas_1"}
                                                    This also defines which columns are items.
                'output_participant_id_col_name' (str, optional): Desired name for PID column in output.
                                                                   Default: 'participant_id'.

        Returns:
            Optional[pd.DataFrame]: DataFrame with participant ID and standardized item columns,
                                    or None if processing fails.
        """
        self.logger.info(f"QuestionnairePreprocessor - Starting item extraction from DataFrame.")

        if input_df is None or input_df.empty:
            self.logger.warning("QuestionnairePreprocessor - Input DataFrame is None or empty. Cannot extract items.")
            return input_df # Return as is, or pd.DataFrame() if preferred for empty

        # Validate essential config keys
        required_keys = ['participant_id_column_original', 'item_column_map']
        for key in required_keys:
            if key not in config:
                self.logger.error(f"QuestionnairePreprocessor - Missing required key in config: '{key}'")
                return None
        if not config['item_column_map']: # Check if the map itself is empty
            self.logger.error("QuestionnairePreprocessor - 'item_column_map' cannot be empty.")
            return None

        # Make a copy to avoid modifying the original DataFrame passed in
        df = input_df.copy()

        # --- "Finding" (selecting and renaming) data ---
        pid_col_original = config['participant_id_column_original']
        item_map = config['item_column_map']
        output_pid_name = config.get('output_participant_id_col_name', 'participant_id')

        if pid_col_original not in df.columns:
            self.logger.error(f"QuestionnairePreprocessor - Participant ID column '{pid_col_original}' not found in the input DataFrame.")
            return None

        columns_to_select_and_rename = {pid_col_original: output_pid_name}
        original_item_columns_from_map = list(item_map.keys())

        missing_original_item_cols = []
        for original_item_col in original_item_columns_from_map:
            if original_item_col not in df.columns:
                missing_original_item_cols.append(original_item_col)
            else:
                # Add to our selection/rename map
                columns_to_select_and_rename[original_item_col] = item_map[original_item_col]

        if missing_original_item_cols:
            self.logger.warning(f"QuestionnairePreprocessor - The following original item columns specified in 'item_column_map' were not found in the input DataFrame: {missing_original_item_cols}")
            # Continue with the ones that were found

        if len(columns_to_select_and_rename) == 1 and item_map: # Only PID was found, but items were expected
            self.logger.error(f"QuestionnairePreprocessor - None of the original item columns from 'item_column_map' were found in the input DataFrame. Cannot proceed.")
            return None

        try:
            # Select only the necessary columns and rename them
            processed_df = df[list(columns_to_select_and_rename.keys())].copy()
            processed_df.rename(columns=columns_to_select_and_rename, inplace=True)
        except KeyError as e:
            self.logger.error(f"QuestionnairePreprocessor - KeyError during column selection/renaming. This might indicate a mismatch not caught earlier: {e}. Columns available: {df.columns.tolist()}")
            return None


        # --- Basic type conversion for items ---
        standard_item_names = [item_map[orig_col] for orig_col in original_item_columns_from_map if orig_col in df.columns]
        for item_col_standard_name in standard_item_names:
            if item_col_standard_name in processed_df.columns: # Check if it made it to processed_df
                try:
                    processed_df[item_col_standard_name] = pd.to_numeric(processed_df[item_col_standard_name], errors='coerce')
                except Exception as e_numeric: # Catch broader errors during conversion
                    self.logger.warning(f"QuestionnairePreprocessor - Could not convert column '{item_col_standard_name}' to numeric. Error: {e_numeric}. Values will be NaN where conversion failed.")
            else:
                # This case should ideally not happen if logic above is correct, but good to be aware
                self.logger.debug(f"QuestionnairePreprocessor - Standard item name '{item_col_standard_name}' not found in processed_df for numeric conversion. This is unexpected.")


        self.logger.info(f"QuestionnairePreprocessor - Successfully extracted questionnaire items. Final shape: {processed_df.shape}")
        return processed_df