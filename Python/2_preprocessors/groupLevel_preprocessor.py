import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
class GroupLevelPreprocessing:
    """
    Prepares group-level data by aggregating information from individual participant artifacts.
    """
    # Default parameters
    DEFAULT_AGGREGATION_TASK_NAME = "Unnamed Aggregation Task"
    DEFAULT_ARTIFACT_DATA_PATH: List[str] = [] # For _concat_dataframes and _collect_metrics
    DEFAULT_COLLECT_METRICS_OUTPUT_COLS: List[str] = ['participant_id', 'metric_name', 'value']


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
        method_name = aggregation_config.get('method') # Orchestrator provides this key directly
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
        artifact_data_path = config.get('artifact_data_path', self.DEFAULT_ARTIFACT_DATA_PATH)
        artifact_data_key = config.get('artifact_data_key')

        if not artifact_data_key:
            self.logger.error(f"Task '{task_name}', ConcatMethod: 'artifact_data_key' missing in config.")
            return None

        temp_dfs: List[pd.DataFrame] = []
        for p_idx, artifact in enumerate(all_participant_artifacts):
            p_id = artifact.get('participant_id', f"P{p_idx:03d}")
            data_location = artifact
            try:
                for key_part in artifact_data_path:
                    data_location = data_location.get(key_part, {})
                df_to_add = data_location.get(artifact_data_key)

                if isinstance(df_to_add, pd.DataFrame) and not df_to_add.empty:
                    df_to_add_copy = df_to_add.copy()
                    if 'participant_id' not in df_to_add_copy.columns:
                        df_to_add_copy['participant_id'] = p_id
                    temp_dfs.append(df_to_add_copy)
                elif df_to_add is not None:
                    self.logger.debug(f"Data for key '{artifact_data_key}' in P:{p_id} is not a non-empty DataFrame (type: {type(df_to_add)}). Skipping.")
            except AttributeError:
                self.logger.debug(f"Path {artifact_data_path} not fully found or invalid in artifact for P:{p_id}, key {artifact_data_key}")

        if temp_dfs:
            current_df = pd.concat(temp_dfs, ignore_index=True)
            self.logger.info(f"Task '{task_name}', ConcatMethod: Aggregated data for key '{artifact_data_key}'. Result shape: {current_df.shape}")
            return current_df
        else:
            self.logger.warning(f"Task '{task_name}', ConcatMethod: No data found to aggregate for key '{artifact_data_key}'.")
            return pd.DataFrame()

    def _collect_metrics(self,
                         all_participant_artifacts: List[Dict[str, Any]],
                         config: Dict[str, Any],
                         task_name: str) -> Optional[pd.DataFrame]:
        artifact_metrics_path = config.get('artifact_metrics_path', self.DEFAULT_ARTIFACT_DATA_PATH)
        metric_definitions = config.get('metric_definitions', [])
        output_columns = config.get('output_columns', self.DEFAULT_COLLECT_METRICS_OUTPUT_COLS)

        collected_rows: List[Dict[str, Any]] = []
        for p_idx, artifact in enumerate(all_participant_artifacts):
            p_id = artifact.get('participant_id', f"P{p_idx:03d}")
            metrics_location = artifact
            try:
                for key_part in artifact_metrics_path:
                    metrics_location = metrics_location.get(key_part, {})
                if not isinstance(metrics_location, dict):
                    self.logger.debug(f"Metrics location at path {artifact_metrics_path} is not a dict for P:{p_id}. Skipping.")
                    continue

                for metric_def in metric_definitions:
                    metric_key = metric_def.get('metric_key')
                    metric_key_template = metric_def.get('metric_key_template')
                    output_metric_name = metric_def.get('output_metric_name')

                    if metric_key: # Direct metric key
                        value = metrics_location.get(metric_key)
                        if value is not None:
                            row_data = {'participant_id': p_id, 'metric_name': output_metric_name or metric_key, 'value': value}
                            for col_name, col_value in metric_def.items(): # Add other fixed columns
                                if col_name not in ['metric_key', 'output_metric_name']:
                                    row_data[col_name] = col_value
                            collected_rows.append(row_data)
                    elif metric_key_template: # Templated metric key
                        iterate_items_key = metric_def.get('iterate_items_from_general_config_key')
                        iteration_list = self.general_configs.get(iterate_items_key, [None]) if iterate_items_key else [None]
                        if iterate_items_key and not self.general_configs.get(iterate_items_key):
                            self.logger.warning(f"Task '{task_name}', CollectMetrics: 'iterate_items_from_general_config_key' '{iterate_items_key}' not found or empty in general_configs.")

                        for item in iteration_list:
                            try:
                                if item is not None:
                                    actual_key = metric_key_template.format(item=item)
                                else:
                                    # Attempt to format without 'item'; if template requires 'item', this might error.
                                    try:
                                        actual_key = metric_key_template.format()
                                    except (KeyError, IndexError): # Template strictly requires 'item'
                                        actual_key = metric_key_template # Use template string as is
                            except Exception as e_format_key:
                                self.logger.warning(f"Task '{task_name}', CollectMetrics: Could not format metric_key_template '{metric_key_template}' with item '{item}'. Error: {e_format_key}. Using template as key.")
                                actual_key = metric_key_template # Fallback

                            value = metrics_location.get(actual_key)
                            if value is not None:
                                row_data = {'participant_id': p_id, 'value': value}
                                out_metric_name_tpl = metric_def.get('output_metric_name_template')
                                if out_metric_name_tpl:
                                    try:
                                        row_data['metric_name'] = out_metric_name_tpl.format(item=item) if item is not None else out_metric_name_tpl.format()
                                    except (KeyError, IndexError, Exception) as e_format_name: # Catch if format fails
                                        self.logger.warning(f"Task '{task_name}', CollectMetrics: Could not format output_metric_name_template '{out_metric_name_tpl}' with item '{item}'. Error: {e_format_name}. Using actual_key or template name.")
                                        row_data['metric_name'] = actual_key # Fallback
                                else:
                                    row_data['metric_name'] = actual_key # Default to actual_key

                                if metric_def.get('output_iteration_column') and item is not None:
                                    row_data[metric_def.get('output_iteration_column')] = item
                                for col_name, col_value in metric_def.items(): # Add other fixed columns
                                    if col_name not in ['metric_key_template', 'iterate_items_from_general_config_key', 'output_iteration_column', 'output_metric_name_template']:
                                        row_data[col_name] = col_value
                                collected_rows.append(row_data)
            except Exception as e_metric_def:
                self.logger.error(f"Error processing metric definition '{metric_def}' for P:{p_id} in task '{task_name}': {e_metric_def}", exc_info=True)

        if collected_rows:
            current_df = pd.DataFrame(collected_rows)
            for col in output_columns: # Ensure all defined output_columns exist
                if col not in current_df.columns: current_df[col] = np.nan
            current_df = current_df[output_columns] # Select and order
            self.logger.info(f"Task '{task_name}', CollectMetrics: Collected metrics. Shape: {current_df.shape}")
            return current_df
        else:
            self.logger.warning(f"Task '{task_name}', CollectMetrics: No metrics collected.")
            return pd.DataFrame(columns=output_columns)