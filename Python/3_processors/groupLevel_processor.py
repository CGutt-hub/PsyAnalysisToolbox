import pandas as pd
from typing import Dict, Any, Optional, List

class GroupLevelProcessing:
    """
    Processes aggregated group-level DataFrames, e.g., by applying filters.
    """
    # Default parameters
    DEFAULT_PROCESSING_TASK_NAME = "Unnamed Processing Task"

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("GroupLevelProcessing initialized.")

    def process_data(self,
                     data_df: pd.DataFrame,
                     processing_config: Dict[str, Any],
                     task_name: str = DEFAULT_PROCESSING_TASK_NAME) -> pd.DataFrame:
        """
        Applies processing steps to the group-level DataFrame.
        Currently supports filtering.

        Args:
            data_df: The aggregated group-level DataFrame to process.
            processing_config: Configuration for processing steps.
            task_name: Name of the task for logging. Defaults to GroupLevelProcessing.DEFAULT_PROCESSING_TASK_NAME.

        Returns:
            The processed DataFrame. Returns the original or an empty DataFrame on issues.
        """
        if data_df is None or data_df.empty:
            self.logger.warning(f"Task '{task_name}': Input DataFrame is None or empty. No processing applied.")
            return data_df if data_df is not None else pd.DataFrame()

        processed_df = data_df.copy()

        # Apply filters if specified
        filter_conditions = processing_config.get('data_filter_conditions') # Orchestrator provides this key
        if filter_conditions and isinstance(filter_conditions, dict):
            self.logger.info(f"Task '{task_name}': Applying data filters: {filter_conditions}.")
            original_rows = len(processed_df)
            try:
                for col, value in filter_conditions.items():
                    if col in processed_df.columns:
                        processed_df = processed_df[processed_df[col].isin(value) if isinstance(value, list) else processed_df[col] == value]
                    else:
                        self.logger.warning(f"Task '{task_name}': Filter column '{col}' not found. Skipping this filter.")
                self.logger.info(f"Task '{task_name}': DataFrame rows after filtering: {len(processed_df)} (was {original_rows}).")
            except Exception as e_filter:
                self.logger.error(f"Task '{task_name}': Error applying filters: {e_filter}", exc_info=True)
                return pd.DataFrame() # Return empty on filter error to signify issue
        return processed_df