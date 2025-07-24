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
                     participant_id_col: Optional[str] = None,
                     trial_id_col: Optional[str] = None,
                     item_id_col: Optional[str] = None,
                     response_value_col: Optional[str] = None,
                     group_by_trial: bool = False
                     ) -> pd.DataFrame:
        """
        Universally scores scales based on provided definitions from a long-format DataFrame.
        Supports both trait (per participant) and state (per participant+trial) scoring.
        All column names and scoring logic are set via config.

        Args:
            data_df (pd.DataFrame): Long-format DataFrame with columns: participant_id, trial_id (optional), item_id, response_value.
            scale_definitions (Dict[str, Dict[str, Any]]):
                A dictionary where keys are scale names and values are dictionaries defining the scale.
            participant_id_col (str): Name of the participant ID column.
            trial_id_col (str): Name of the trial/scene/epoch column (optional).
            item_id_col (str): Name of the item/question ID column.
            response_value_col (str): Name of the response value column.
            group_by_trial (bool): If True, score per participant+trial (state); if False, per participant (trait).
        Returns:
            pd.DataFrame: DataFrame with index [participant_id] or [participant_id, trial_id] and columns for each scale.
        """
        if data_df is None or data_df.empty:
            self.logger.warning("QuestionnaireScaleProcessor - Input DataFrame is empty or None. Cannot score scales.")
            return pd.DataFrame()
        if not scale_definitions:
            self.logger.warning("QuestionnaireScaleProcessor - No scale definitions provided. Returning original data.")
            return data_df.copy()
        # Set column names
        pid_col = participant_id_col or 'participant_id'
        trial_col = trial_id_col or 'trial_id'
        item_col = item_id_col or 'item_id'
        resp_col = response_value_col or 'response_value'
        # Determine groupby columns
        group_cols = [pid_col]
        if group_by_trial and trial_col in data_df.columns:
            group_cols.append(trial_col)
        # Prepare output DataFrame
        scores_df = pd.DataFrame()
        self.logger.info(f"QuestionnaireScaleProcessor - Starting universal scale scoring for {len(scale_definitions)} scales.")
        for scale_name, definition in scale_definitions.items():
            self.logger.debug(f"QuestionnaireScaleProcessor - Scoring scale: {scale_name}")
            items = definition.get(self.KEY_ITEMS, [])
            scoring_method = definition.get(self.KEY_SCORING_METHOD, self.DEFAULT_SCORING_METHOD).lower()
            reverse_coded = definition.get(self.KEY_REVERSE_CODED_ITEMS, {})
            min_valid_ratio = definition.get(self.KEY_MIN_VALID_ITEMS_RATIO)
            if not items:
                self.logger.warning(f"QuestionnaireScaleProcessor - No items defined for scale '{scale_name}'. Skipping.")
                continue
            # Filter for relevant items
            df_items = data_df[data_df[item_col].isin(items)].copy()
            # Reverse code if needed
            for item in items:
                if item in reverse_coded:
                    rc_def = reverse_coded[item]
                    mask = df_items[item_col] == item
                    df_items.loc[mask, resp_col] = self._reverse_code_item(df_items.loc[mask, resp_col], rc_def[self.KEY_RC_MIN_VAL], rc_def[self.KEY_RC_MAX_VAL])
            # Group and score
            def score_group(grp):
                vals = grp[resp_col]
                valid = vals.notna().sum()
                if min_valid_ratio is not None:
                    if valid < np.ceil(len(items) * min_valid_ratio):
                        return np.nan
                if scoring_method == "sum":
                    return vals.sum(skipna=True)
                elif scoring_method == "mean":
                    return vals.mean(skipna=True)
                else:
                    self.logger.warning(f"QuestionnaireScaleProcessor - Unknown scoring method '{scoring_method}' for scale '{scale_name}'. Returning NaN.")
                    return np.nan
            scale_scores = df_items.groupby(group_cols).apply(score_group)
            scale_scores.name = scale_name
            scores_df = pd.concat([scores_df, scale_scores], axis=1)
        scores_df.reset_index(inplace=True)
        self.logger.info("QuestionnaireScaleProcessor - Universal scale scoring completed.")
        return scores_df