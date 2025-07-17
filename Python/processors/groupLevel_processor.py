import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class GroupLevelProcessor:
    """
    Prepares and processes group-level data by aggregating information from individual
    participant artifacts and applying subsequent transformations like filtering.
    """
    # Class-level defaults
    DEFAULT_AGGREGATION_TASK_NAME = "Unnamed Aggregation Task"
    DEFAULT_PROCESSING_TASK_NAME = "Unnamed Processing Task"
    DEFAULT_ARTIFACT_DATA_PATH: List[str] = []
    DEFAULT_COLLECT_METRICS_OUTPUT_COLS: List[str] = ['participant_id', 'metric_name', 'value']

    # --- Configuration Keys ---
    # For dispatching
    CONFIG_KEY_METHOD = 'method'
    # For _concat_dataframes
    CONFIG_KEY_ARTIFACT_DATA_PATH = 'artifact_data_path'
    CONFIG_KEY_ARTIFACT_DATA_KEY = 'artifact_data_key'
    # For _collect_metrics
    CONFIG_KEY_ARTIFACT_METRICS_PATH = 'artifact_metrics_path'
    CONFIG_KEY_METRIC_DEFINITIONS = 'metric_definitions'
    CONFIG_KEY_OUTPUT_COLUMNS = 'output_columns'
    # For data processing
    CONFIG_KEY_DATA_FILTER_CONDITIONS = 'data_filter_conditions'
    # --- Internal Keys for Metric Definitions ---
    METRIC_DEF_KEY_METRIC_KEY = 'metric_key'
    METRIC_DEF_KEY_METRIC_KEY_TEMPLATE = 'metric_key_template'
    METRIC_DEF_KEY_OUTPUT_METRIC_NAME = 'output_metric_name'
    METRIC_DEF_KEY_ITERATE_ITEMS_KEY = 'iterate_items_from_general_config_key'
    METRIC_DEF_KEY_OUTPUT_ITERATION_COLUMN = 'output_iteration_column'
    METRIC_DEF_KEY_OUTPUT_METRIC_NAME_TEMPLATE = 'output_metric_name_template'

    def __init__(self, logger, general_configs: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.general_configs = general_configs if general_configs is not None else {}
        self.logger.info("GroupLevelProcessing initialized.")

    def aggregate_data(self,
                       all_participant_artifacts: List[Dict[str, Any]],
                       aggregation_config: Dict[str, Any],
                       task_name: str = DEFAULT_AGGREGATION_TASK_NAME) -> Optional[pd.DataFrame]:
        """
        Dispatches to the appropriate aggregation method based on the configuration.
        """
        method_name = aggregation_config.get(self.CONFIG_KEY_METHOD)
        self.logger.info(f"Task '{task_name}': Attempting aggregation method '{method_name}'.")

        if method_name == "concat_dataframes_from_artifacts":
            return self._concat_dataframes(all_participant_artifacts, aggregation_config, task_name)
        elif method_name == "collect_metrics_to_dataframe":
            return self._collect_metrics(all_participant_artifacts, aggregation_config, task_name)
        else:
            self.logger.error(f"Task '{task_name}': Unknown aggregation method '{method_name}'.")
            return None

    def process_data(self,
                     data_df: pd.DataFrame,
                     processing_config: Dict[str, Any],
                     task_name: str = DEFAULT_PROCESSING_TASK_NAME) -> pd.DataFrame:
        """
        Applies processing steps like filtering to an aggregated group-level DataFrame.
        """
        if data_df is None or data_df.empty:
            self.logger.warning(f"Task '{task_name}': Input DataFrame is None or empty. No processing applied.")
            return data_df if data_df is not None else pd.DataFrame()

        processed_df = data_df.copy()

        filter_conditions = processing_config.get(self.CONFIG_KEY_DATA_FILTER_CONDITIONS)
        if filter_conditions and isinstance(filter_conditions, dict):
            self.logger.info(f"Task '{task_name}': Applying data filters: {filter_conditions}.")
            original_rows = len(processed_df)
            try:
                for col, value in filter_conditions.items():
                    if col in processed_df.columns:
                        is_list = isinstance(value, list)
                        processed_df = processed_df[processed_df[col].isin(value) if is_list else (processed_df[col] == value)]
                    else:
                        self.logger.warning(f"Task '{task_name}': Filter column '{col}' not found. Skipping filter.")
                self.logger.info(f"Task '{task_name}': Rows after filtering: {len(processed_df)} (from {original_rows}).")
            except Exception as e:
                self.logger.error(f"Task '{task_name}': Error applying filters: {e}", exc_info=True)
                return pd.DataFrame()
        return processed_df

    def _get_nested_data(self, artifact_dict: Dict, path: List[str]) -> Any:
        """Safely traverses a nested dictionary path."""
        data = artifact_dict
        for key in path:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data

    def _concat_dataframes(self, all_artifacts: List[Dict], config: Dict, task_name: str) -> Optional[pd.DataFrame]:
        """Concatenates DataFrames found at a specified path in each participant's artifacts."""
        path = config.get(self.CONFIG_KEY_ARTIFACT_DATA_PATH, [])
        key = config.get(self.CONFIG_KEY_ARTIFACT_DATA_KEY)
        if not key:
            self.logger.error(f"Task '{task_name}': 'artifact_data_key' is missing for concat method.")
            return None

        dfs_to_concat = []
        for p_artifact in all_artifacts:
            p_id = p_artifact.get('participant_id', 'Unknown_PID')
            data_location = self._get_nested_data(p_artifact, path)
            if isinstance(data_location, dict):
                df_to_add = data_location.get(key)
                if isinstance(df_to_add, pd.DataFrame) and not df_to_add.empty:
                    df_copy = df_to_add.copy()
                    if 'participant_id' not in df_copy.columns:
                        df_copy['participant_id'] = p_id
                    dfs_to_concat.append(df_copy)

        if dfs_to_concat:
            result_df = pd.concat(dfs_to_concat, ignore_index=True)
            self.logger.info(f"Task '{task_name}': Concatenated {len(dfs_to_concat)} DataFrames. Shape: {result_df.shape}")
            return result_df
        else:
            self.logger.warning(f"Task '{task_name}': No valid DataFrames found at path '{path}' with key '{key}'.")
            return pd.DataFrame()

    def _collect_metrics(self, all_artifacts: List[Dict], config: Dict, task_name: str) -> Optional[pd.DataFrame]:
        """Collects scalar metrics from each participant's artifacts into a long-format DataFrame."""
        path = config.get(self.CONFIG_KEY_ARTIFACT_METRICS_PATH, [])
        definitions = config.get(self.CONFIG_KEY_METRIC_DEFINITIONS, [])
        output_cols = config.get(self.CONFIG_KEY_OUTPUT_COLUMNS, self.DEFAULT_COLLECT_METRICS_OUTPUT_COLS)

        collected_rows = []
        for p_artifact in all_artifacts:
            p_id = p_artifact.get('participant_id', 'Unknown_PID')
            metrics_location = self._get_nested_data(p_artifact, path)
            if not isinstance(metrics_location, dict):
                continue

            for metric_def in definitions:
                if not isinstance(metric_def, dict): continue

                metric_key_template = metric_def.get(self.METRIC_DEF_KEY_METRIC_KEY_TEMPLATE)
                iterate_items_key = metric_def.get(self.METRIC_DEF_KEY_ITERATE_ITEMS_KEY)

                iteration_list = self.general_configs.get(iterate_items_key, [None]) if iterate_items_key else [None]

                for item in iteration_list:
                    try:
                        actual_key = metric_key_template.format(item=item) if item is not None and isinstance(metric_key_template, str) else metric_key_template
                        value = metrics_location.get(actual_key)

                        if value is not None:
                            row_data = {'participant_id': p_id, 'value': value}
                            # Add metric name
                            name_template = metric_def.get(self.METRIC_DEF_KEY_OUTPUT_METRIC_NAME_TEMPLATE, actual_key)
                            row_data['metric_name'] = name_template.format(item=item) if item is not None and isinstance(name_template, str) else name_template
                            # Add iteration item as a column
                            iter_col_name = metric_def.get(self.METRIC_DEF_KEY_OUTPUT_ITERATION_COLUMN)
                            if iter_col_name and item is not None:
                                row_data[iter_col_name] = item
                            
                            collected_rows.append(row_data)
                    except Exception as e:
                        self.logger.warning(f"Task '{task_name}': Error processing metric definition {metric_def} for P:{p_id}. Error: {e}")

        if collected_rows:
            result_df = pd.DataFrame(collected_rows)
            # Ensure all defined output columns exist, even if they weren't populated
            for col in output_cols:
                if col not in result_df.columns:
                    result_df[col] = np.nan
            result_df = result_df[output_cols] # Select and order columns
            self.logger.info(f"Task '{task_name}': Collected {len(result_df)} metrics. Shape: {result_df.shape}")
            return result_df
        else:
            self.logger.warning(f"Task '{task_name}': No metrics collected.")
            return pd.DataFrame(columns=output_cols)
