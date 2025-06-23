import pandas as pd
from typing import Dict, Any, Optional, List

class GroupLevelProcessing:
    """
    Processes aggregated group-level DataFrames, e.g., by applying filters.
    """
    # Default parameters
    DEFAULT_PROCESSING_TASK_NAME = "Unnamed Processing Task"
    CONFIG_KEY_DATA_FILTER_CONDITIONS = 'data_filter_conditions'

    def __init__(self, logger, general_configs: Dict[str, str]):
        self.logger = logger
        self.general_configs = general_configs # Store general configs if needed by aggregation methods
        self.logger.info("GroupLevelProcessing initialized.")

    def aggregate_data(self,
                       all_participant_artifacts: List[Dict[str, Any]],
                       aggregation_config: Dict[str, Any],
                       task_name: str = "Aggregation Task") -> Optional[pd.DataFrame]:
        """
        Aggregates data from multiple participant artifacts based on configuration.

        Args:
            all_participant_artifacts: A list of dictionaries, where each dictionary
                                       contains artifacts for one participant.
            aggregation_config: Configuration for the aggregation step, e.g.,
                                {'method': 'concat_dataframes_from_artifacts',
                                 'artifact_data_path': 'plv_results_df'}.
            task_name: Name of the aggregation task for logging.

        Returns:
            A pandas DataFrame containing the aggregated data, or None if aggregation fails.
        """
        method = aggregation_config.get('method')
        artifact_data_path = aggregation_config.get('artifact_data_path')

        if not all_participant_artifacts:
            self.logger.warning(f"Aggregation task '{task_name}': No participant artifacts provided.")
            return None
        if not method:
            self.logger.error(f"Aggregation task '{task_name}': No aggregation method specified in config.")
            return None
        if not artifact_data_path:
            self.logger.error(f"Aggregation task '{task_name}': No 'artifact_data_path' specified for method '{method}'.")
            return None

        aggregated_df = None
        if method == 'concat_dataframes_from_artifacts':
            dfs_to_concat = []
            for p_artifacts in all_participant_artifacts:
                current_artifact = p_artifacts.get(artifact_data_path) # Direct access for now, can be extended for nested paths
                if isinstance(current_artifact, pd.DataFrame) and not current_artifact.empty:
                    dfs_to_concat.append(current_artifact)
            if dfs_to_concat:
                aggregated_df = pd.concat(dfs_to_concat, ignore_index=True)
                self.logger.info(f"Aggregation task '{task_name}': Concatenated {len(dfs_to_concat)} DataFrames. Result shape: {aggregated_df.shape}")
            else:
                self.logger.warning(f"Aggregation task '{task_name}': No valid DataFrames found to concatenate.")
        else:
            self.logger.error(f"Aggregation task '{task_name}': Unknown aggregation method: '{method}'.")

        return aggregated_df

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

        if not isinstance(processing_config, dict):
            self.logger.warning(f"Task '{task_name}': 'processing_config' is not a dictionary (type: {type(processing_config)}). "
                                f"No processing applied, returning original DataFrame.")
            return data_df # Return original df if config is invalid

        processed_df = data_df.copy()

        # Apply filters if specified
        filter_conditions = processing_config.get(self.CONFIG_KEY_DATA_FILTER_CONDITIONS)
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