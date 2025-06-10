import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Module-level defaults for GroupLevelPreprocessing
DEFAULT_AGGREGATION_TASK_NAME = "Unnamed Aggregation Task"
DEFAULT_ARTIFACT_DATA_PATH: List[str] = [] # For _concat_dataframes and _collect_metrics
DEFAULT_COLLECT_METRICS_OUTPUT_COLS: List[str] = ['participant_id', 'metric_name', 'value']

class GroupLevelPreprocessing:
    """
    Prepares group-level data by aggregating information from individual participant artifacts.
    """

    def __init__(self, logger, general_configs: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.general_configs = general_configs if general_configs is not None else {}
        self.logger.info("GroupLevelPreprocessing initialized.")

    def aggregate_data(self,
                       all_participant_artifacts: List[Dict[str, Any]],
                       aggregation_config: Dict[str, Any],
                       task_name: str = DEFAULT_AGGREGATION_TASK_NAME) -> Optional[pd.DataFrame]:
        """
        Dispatches to the appropriate aggregation method based on the configuration.

        Args:
            all_participant_artifacts: List of all participant artifact dicts.
            aggregation_config: Configuration specific to this aggregation task.
                                Must contain a 'method' key.
            task_name: Name of the task for logging.

        Returns:
            Aggregated DataFrame or None on failure.
        """
        method_name = aggregation_config.get('method')
        self.logger.info(f"Task '{task_name}': Attempting aggregation method '{method_name}'.")

        if method_name == "concat_dataframes_from_artifacts":
            return self._concat_dataframes(all_participant_artifacts, aggregation_config, task_name)
        elif method_name == "collect_metrics_to_dataframe":
             return self._collect_metrics(all_participant_artifacts, aggregation_config, task_name)
        # Add other elif for new aggregation methods here
        else:
            self.logger.error(f"Task '{task_name}': Unknown aggregation method '{method_name}'.")
            return None

    def _concat_dataframes(self,
                           all_participant_artifacts: List[Dict[str, Any]],
                           config: Dict[str, Any],
                           task_name: str) -> Optional[pd.DataFrame]:
        
        # Validate and get artifact_data_path
        artifact_data_path_config = config.get('artifact_data_path')
        final_artifact_data_path: List[str] = DEFAULT_ARTIFACT_DATA_PATH
        if artifact_data_path_config is not None:
            if isinstance(artifact_data_path_config, list) and all(isinstance(item, str) for item in artifact_data_path_config):
                final_artifact_data_path = artifact_data_path_config
            else:
                self.logger.warning(f"Task '{task_name}', ConcatMethod: Invalid 'artifact_data_path' ('{artifact_data_path_config}'). Expected List[str]. Using default: {DEFAULT_ARTIFACT_DATA_PATH}.")

        # Validate and get artifact_data_key
        artifact_data_key = config.get('artifact_data_key')
        if not isinstance(artifact_data_key, str) or not artifact_data_key.strip():
            self.logger.error(f"Task '{task_name}', ConcatMethod: 'artifact_data_key' missing in config.")
            return None
        final_artifact_data_key = artifact_data_key.strip()

        temp_dfs: List[pd.DataFrame] = []
        for p_idx, artifact in enumerate(all_participant_artifacts):
            p_id = artifact.get('participant_id', f"P{p_idx:03d}")
            data_location = artifact
            try:
                for key_part in final_artifact_data_path:
                    data_location = data_location.get(key_part, {})
                df_to_add = data_location.get(final_artifact_data_key)

                if isinstance(df_to_add, pd.DataFrame) and not df_to_add.empty:
                    df_to_add_copy = df_to_add.copy()
                    if 'participant_id' not in df_to_add_copy.columns:
                        df_to_add_copy['participant_id'] = p_id
                    temp_dfs.append(df_to_add_copy)
                elif df_to_add is not None: # Log if key exists but isn't a non-empty DataFrame
                    self.logger.debug(f"Task '{task_name}', ConcatMethod: Data for key '{final_artifact_data_key}' in P:{p_id} is not a non-empty DataFrame (type: {type(df_to_add)}). Skipping.")
            except AttributeError:
                self.logger.debug(f"Path {final_artifact_data_path} not fully found or invalid in artifact for P:{p_id}, key {final_artifact_data_key}")

        if temp_dfs:
            current_df = pd.concat(temp_dfs, ignore_index=True)
            self.logger.info(f"Task '{task_name}', ConcatMethod: Aggregated data for key '{final_artifact_data_key}'. Result shape: {current_df.shape}")
            return current_df
        else:
            self.logger.warning(f"Task '{task_name}', ConcatMethod: No data found to aggregate for key '{final_artifact_data_key}'.")
            return pd.DataFrame()


    def _collect_metrics(self,
                         all_participant_artifacts: List[Dict[str, Any]],
                         config: Dict[str, Any],
                         task_name: str) -> Optional[pd.DataFrame]:
        
        # Validate and get artifact_metrics_path
        artifact_metrics_path_config = config.get('artifact_metrics_path')
        final_artifact_metrics_path: List[str] = DEFAULT_ARTIFACT_DATA_PATH # Use the same default list type
        if artifact_metrics_path_config is not None:
            if isinstance(artifact_metrics_path_config, list) and all(isinstance(item, str) for item in artifact_metrics_path_config):
                final_artifact_metrics_path = artifact_metrics_path_config
            else:
                self.logger.warning(f"Task '{task_name}', CollectMetrics: Invalid 'artifact_metrics_path' ('{artifact_metrics_path_config}'). Expected List[str]. Using default: {DEFAULT_ARTIFACT_DATA_PATH}.")

        # Validate and get metric_definitions
        metric_definitions_config = config.get('metric_definitions')
        if not isinstance(metric_definitions_config, list):
             self.logger.error(f"Task '{task_name}', CollectMetrics: 'metric_definitions' missing or invalid in config. Expected List.")
             return None
        final_metric_definitions: List[Dict[str, Any]] = metric_definitions_config # Assume list of dicts based on expected structure

        # Validate and get output_columns
        output_columns_config = config.get('output_columns')
        final_output_columns: List[str] = DEFAULT_COLLECT_METRICS_OUTPUT_COLS
        if output_columns_config is not None:
            if isinstance(output_columns_config, list) and all(isinstance(item, str) and item.strip() for item in output_columns_config):
                final_output_columns = [col.strip() for col in output_columns_config]
            else:
                self.logger.warning(f"Task '{task_name}', CollectMetrics: Invalid 'output_columns' ('{output_columns_config}'). Expected List[str]. Using default: {DEFAULT_COLLECT_METRICS_OUTPUT_COLS}.")

        collected_rows: List[Dict[str, Any]] = []
        for p_idx, artifact in enumerate(all_participant_artifacts):
            p_id = artifact.get('participant_id', f"P{p_idx:03d}")
            metrics_location = artifact
            try:
                for key_part in final_artifact_metrics_path:
                    metrics_location = metrics_location.get(key_part, {})
                if not isinstance(metrics_location, dict):
                    self.logger.debug(f"Metrics location at path {final_artifact_metrics_path} is not a dict for P:{p_id}. Skipping.")
                    continue

                for i, metric_def in enumerate(final_metric_definitions):
                    # Ensure metric_def is defined in this scope
                    metric_key = metric_def.get('metric_key')
                    metric_key_template = metric_def.get('metric_key_template')
                    output_metric_name = metric_def.get('output_metric_name')

                    # Validate metric definition structure
                    is_direct_key = isinstance(metric_key, str) and metric_key.strip()
                    is_template_key = isinstance(metric_key_template, str) and metric_key_template.strip()

                    if not is_direct_key and not is_template_key:
                        self.logger.warning(f"Task '{task_name}', CollectMetrics: Metric definition {i} for P:{p_id} is missing both 'metric_key' and 'metric_key_template' or they are invalid. Skipping definition.")
                        continue # Skip this invalid definition

                    if is_direct_key: # Direct metric key
                        value = metrics_location.get(metric_key)
                        if value is not None:
                            row_data = {'participant_id': p_id, 'metric_name': output_metric_name or metric_key, 'value': value}
                            for col_name, col_value in metric_def.items(): # Add other fixed columns
                                if col_name not in ['metric_key', 'output_metric_name']:
                                    row_data[col_name] = col_value
                            collected_rows.append(row_data)
                    elif is_template_key: # Templated metric key
                        iterate_items_key = metric_def.get('iterate_items_from_general_config_key')
                        iteration_list = self.general_configs.get(iterate_items_key, [None]) if iterate_items_key else [None]
                        if iterate_items_key and not self.general_configs.get(iterate_items_key):
                            self.logger.warning(f"Task '{task_name}', CollectMetrics: 'iterate_items_from_general_config_key' '{iterate_items_key}' not found or empty in general_configs.")

                        for item in iteration_list:
                            try:
                                if item is not None:
                                    # Ensure item is string-like for formatting, handle None gracefully
                                    item_str = str(item) if item is not None else ""
                                    actual_key = metric_key_template.format(item=item_str)
                                else:
                                    # Attempt to format without 'item'; if template requires 'item', this might error.
                                    try:
                                        actual_key = metric_key_template.format()
                                    except (KeyError, IndexError, AttributeError): # Template strictly requires 'item' or is not a string
                                        actual_key = metric_key_template # Use template string as is
                            except Exception as e_format_key:
                                self.logger.warning(f"Task '{task_name}', CollectMetrics: Could not format metric_key_template '{metric_key_template}' with item '{item}'. Error: {e_format_key}. Using template as key.")
                                actual_key = metric_key_template # Fallback

                            value = metrics_location.get(actual_key)
                            if value is not None and isinstance(actual_key, str): # Ensure actual_key is a string before using it
                                row_data = {'participant_id': p_id, 'value': value}
                                out_metric_name_tpl = metric_def.get('output_metric_name_template')
                                if isinstance(out_metric_name_tpl, str) and out_metric_name_tpl.strip():
                                    try: # Ensure out_metric_name_tpl is a string before calling format
                                         # Ensure item is string-like for formatting, handle None gracefully
                                        item_str = str(item) if item is not None else ""
                                        row_data['metric_name'] = out_metric_name_tpl.format(item=item_str) if item is not None else out_metric_name_tpl.format()
                                    except (KeyError, IndexError, Exception) as e_format_name: # Catch if format fails
                                        self.logger.warning(f"Task '{task_name}', CollectMetrics: Could not format output_metric_name_template '{out_metric_name_tpl}' with item '{item}'. Error: {e_format_name}. Using actual_key or template name.")
                                        row_data['metric_name'] = actual_key # Fallback
                                else:
                                    row_data['metric_name'] = actual_key # Default to actual_key

                                # Validate and add output_iteration_column
                                output_iter_col_name = metric_def.get('output_iteration_column')
                                if isinstance(output_iter_col_name, str) and output_iter_col_name.strip() and item is not None:
                                    row_data[output_iter_col_name] = item
                                for col_name, col_value in metric_def.items(): # Add other fixed columns
                                    if col_name not in ['metric_key_template', 'iterate_items_from_general_config_key', 'output_iteration_column', 'output_metric_name_template']:
                                        row_data[col_name] = col_value
                                collected_rows.append(row_data)
            except Exception as e_metric_def:
                self.logger.error(f"Error processing metric definition '{metric_def}' for P:{p_id} in task '{task_name}': {e_metric_def}", exc_info=True)

        if collected_rows:
            current_df = pd.DataFrame(collected_rows)
            for col in final_output_columns: # Ensure all defined output_columns exist
                if col not in current_df.columns: current_df[col] = np.nan
            current_df = current_df[final_output_columns] # Select and order
            self.logger.info(f"Task '{task_name}', CollectMetrics: Collected metrics. Shape: {current_df.shape}")
            return current_df
        else:
            self.logger.warning(f"Task '{task_name}', CollectMetrics: No metrics collected.")
            return pd.DataFrame(columns=final_output_columns)