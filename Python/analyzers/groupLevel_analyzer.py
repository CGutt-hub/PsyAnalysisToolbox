"""
Group Level Analyzer Module
--------------------------
Performs group-level analyses on aggregated participant data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class GroupLevelAnalyzer:
    """
    Performs group-level analyses on aggregated participant data.
    - Accepts config dict for analysis parameters.
    - Returns group-level results as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("GroupLevelAnalyzer initialized.")

    def analyze(self, group_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Performs group-level analysis on the provided data using config parameters.
        Returns a DataFrame with group-level results.
        """
        self.logger.info("GroupLevelAnalyzer: Performing group-level analysis (mean and std for all numeric columns).")
        if group_data is None or group_data.empty:
            self.logger.warning("GroupLevelAnalyzer: No group data provided. Returning empty DataFrame.")
            return pd.DataFrame()
        numeric_cols = group_data.select_dtypes(include='number').columns
        means = group_data[numeric_cols].mean()
        stds = group_data[numeric_cols].std()
        result = pd.DataFrame({'metric': list(means.index) + list(stds.index),
                               'stat': ['mean'] * len(means) + ['std'] * len(stds),
                               'value': list(means.values) + list(stds.values)})
        return result