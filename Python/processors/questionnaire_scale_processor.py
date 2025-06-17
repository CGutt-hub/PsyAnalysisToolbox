import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

class QuestionnaireScaleProcessor:
    # Default parameters for scale scoring
    DEFAULT_SCORING_METHOD = "sum"

    # Keys for scale_definitions dictionary
    KEY_ITEMS = "items"
    KEY_SCORING_METHOD = "scoring_method"
    KEY_REVERSE_CODED_ITEMS = "reverse_coded_items"
    KEY_RC_MIN_VAL = "min_val" # For sub-dictionary in reverse_coded_items
    KEY_RC_MAX_VAL = "max_val" # For sub-dictionary in reverse_coded_items
    KEY_MIN_VALID_ITEMS_RATIO = "min_valid_items_ratio"

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("QuestionnaireScaleProcessor initialized.")

    def _reverse_code_item(self, series: pd.Series, min_val: Union[int, float], max_val: Union[int, float]) -> pd.Series: # type: ignore
        """Helper to reverse code a single item (pandas Series).
           Assumes 'series' has already been coerced to numeric if necessary."""
        # 'series' is expected to be numeric (or contain NaNs from prior coercion)
        return (max_val + min_val) - series

    def score_scales(self,
                     data_df: pd.DataFrame,
                     scale_definitions: Dict[str, Dict[str, Any]],
                     participant_id_col: Optional[str] = None
                     ) -> pd.DataFrame:
        """
        Scores scales based on provided definitions.

        Args:
            data_df (pd.DataFrame): DataFrame containing the raw item data.
                                    If participant_id_col is provided and exists in data_df,
                                    it will be included in the output DataFrame.
            scale_definitions (Dict[str, Dict[str, Any]]):
                A dictionary where keys are scale names and values are dictionaries
                defining the scale. Example definition for a scale:
                {
                    "items": ["item1", "item2", "item3_r"], // List of column names for items
                    "scoring_method": "sum", // "sum" or "mean"
                    "reverse_coded_items": { // Optional
                        "item3_r": {"min_val": 1, "max_val": 5} // item_name: {min_val, max_val}
                    },
                    "min_valid_items_ratio": 0.8 // Optional: float (0.0 to 1.0). If provided,
                                                 // scale score is NaN if fewer valid items.
                }
            participant_id_col (Optional[str]): Name of the column containing participant IDs.
                                                 If provided and exists in data_df, it will be
                                                 included in the output DataFrame alongside scores.
        Returns:
            pd.DataFrame: A DataFrame with original data (if participant_id_col is used to re-index)
                          or just participant IDs as index, and new columns for each calculated scale score.
                          If participant_id_col is None, it returns a DataFrame with only scores and original index.
        """
        if data_df is None or data_df.empty:
            self.logger.warning("QuestionnaireScaleProcessor - Input DataFrame is empty or None. Cannot score scales.")
            return pd.DataFrame()

        if not scale_definitions:
            self.logger.warning("QuestionnaireScaleProcessor - No scale definitions provided. Returning original data.")
            return data_df.copy()

        # Prepare the output DataFrame to hold scores
        # It will initially have the same index as the input data_df
        scores_df = pd.DataFrame(index=data_df.index)

        # Determine if participant ID column should be included in the final output
        include_pid_in_output = participant_id_col and participant_id_col in data_df.columns
        if participant_id_col and participant_id_col in data_df.columns:
            self.logger.debug(f"QuestionnaireScaleProcessor - Participant ID column '{participant_id_col}' found. Will include in output.")

        self.logger.info(f"QuestionnaireScaleProcessor - Starting scale scoring for {len(scale_definitions)} scales.")

        for scale_name, definition in scale_definitions.items():
            self.logger.debug(f"QuestionnaireScaleProcessor - Scoring scale: {scale_name}")
            items = definition.get(self.KEY_ITEMS, [])
            scoring_method = definition.get(self.KEY_SCORING_METHOD, self.DEFAULT_SCORING_METHOD).lower()
            reverse_coded = definition.get(self.KEY_REVERSE_CODED_ITEMS, {})
            min_valid_ratio = definition.get(self.KEY_MIN_VALID_ITEMS_RATIO)

            if not items:
                self.logger.warning(f"QuestionnaireScaleProcessor - No items defined for scale '{scale_name}'. Skipping.")
                continue

            # Select item columns and handle reverse coding
            scale_item_data_list = []
            for item_col in items:
                if item_col not in data_df.columns: # Use original data_df for column check
                    self.logger.warning(f"QuestionnaireScaleProcessor - Item '{item_col}' for scale '{scale_name}' not found in data. Skipping item.")
                    scale_item_data_list.append(pd.Series(np.nan, index=data_df.index)) # Add NaNs for this item
                    continue
                
                item_series = pd.to_numeric(data_df[item_col], errors='coerce') # Use original data_df for data
                if item_col in reverse_coded:
                    rc_def = reverse_coded[item_col]
                    item_series = self._reverse_code_item(item_series, rc_def[self.KEY_RC_MIN_VAL], rc_def[self.KEY_RC_MAX_VAL])
                scale_item_data_list.append(item_series)
            
            if not scale_item_data_list: # All items were missing
                self.logger.warning(f"QuestionnaireScaleProcessor - No valid items found for scale '{scale_name}' after checking columns. Skipping scale.")
                scores_df[scale_name] = np.nan
                continue

            scale_df_temp = pd.concat(scale_item_data_list, axis=1)
            
            # Calculate score based on method
            if scoring_method == "sum":
                raw_scores = scale_df_temp.sum(axis=1, skipna=True) # skipna=True sums available data
            elif scoring_method == "mean":
                raw_scores = scale_df_temp.mean(axis=1, skipna=True)
            else:
                self.logger.warning(f"QuestionnaireScaleProcessor - Unknown scoring method '{scoring_method}' for scale '{scale_name}'. Skipping.")
                scores_df[scale_name] = np.nan
                continue
            
            # Handle minimum valid items ratio
            if min_valid_ratio is not None:
                valid_counts = scale_df_temp.notna().sum(axis=1)
                required_items = np.ceil(len(items) * min_valid_ratio) # type: ignore
                raw_scores[valid_counts < required_items] = np.nan

            scores_df[scale_name] = raw_scores
            self.logger.debug(f"QuestionnaireScaleProcessor - Finished scoring scale: {scale_name}")

        self.logger.info("QuestionnaireScaleProcessor - Scale scoring completed.")
        
        # Combine participant ID column (if requested) with scores
        final_output_df = scores_df
        if include_pid_in_output:
            # Create a DataFrame with just the PID column from the original data_df
            pid_df = data_df[[participant_id_col]].copy()
            # Concatenate PID column with the scores_df (which has the same index)
            final_output_df = pd.concat([pid_df, scores_df], axis=1)
            self.logger.debug(f"QuestionnaireScaleProcessor - Combined PID column '{participant_id_col}' with scores.")

        return final_output_df