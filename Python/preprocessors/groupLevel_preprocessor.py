import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional

class GroupLevelPreprocessing:
    """
    Prepares group-level data by aggregating information from individual participant artifacts.
    """
    # Class-level defaults
    DEFAULT_AGGREGATION_TASK_NAME = "Unnamed Aggregation Task"
    DEFAULT_ARTIFACT_DATA_PATH: List[str] = [] # For _concat_dataframes and _collect_metrics
    DEFAULT_COLLECT_METRICS_OUTPUT_COLS: List[str] = ['participant_id', 'metric_name', 'value']

    # Configuration keys
    CONFIG_KEY_METHOD = 'method'
    CONFIG_KEY_ARTIFACT_DATA_PATH = 'artifact_data_path' # Used by _concat_dataframes
    CONFIG_KEY_ARTIFACT_DATA_KEY = 'artifact_data_key'   # Used by _concat_dataframes
    CONFIG_KEY_ARTIFACT_METRICS_PATH = 'artifact_metrics_path' # Used by _collect_metrics
    CONFIG_KEY_METRIC_DEFINITIONS = 'metric_definitions'     # Used by _collect_metrics
    CONFIG_KEY_OUTPUT_COLUMNS = 'output_columns'             # Used by _collect_metrics
    # Metric definition internal keys
    METRIC_DEF_KEY_METRIC_KEY = 'metric_key'
    METRIC_DEF_KEY_METRIC_KEY_TEMPLATE = 'metric_key_template'
    METRIC_DEF_KEY_OUTPUT_METRIC_NAME = 'output_metric_name'
    METRIC_DEF_KEY_ITERATE_ITEMS_KEY = 'iterate_items_from_general_config_key'
    METRIC_DEF_KEY_OUTPUT_ITERATION_COLUMN = 'output_iteration_column'
    METRIC_DEF_KEY_OUTPUT_METRIC_NAME_TEMPLATE = 'output_metric_name_template'

    def __init__(self, logger: logging.Logger, general_configs: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.general_configs = general_configs if general_configs is not None else {}
        self.logger.info("GroupLevelPreprocessing initialized.")

    def aggregate_data(self,
                       all_participant_artifacts: List[Dict[str, Any]],
                       aggregation_config: Dict[str, Any],
                       task_name: str = DEFAULT_AGGREGATION_TASK_NAME) -> Optional[pd.DataFrame]: # Can use self.DEFAULT_... if instance method
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
        method_name = aggregation_config.get(self.CONFIG_KEY_METHOD)
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
        artifact_data_path_config = config.get(self.CONFIG_KEY_ARTIFACT_DATA_PATH)
        final_artifact_data_path: List[str] = self.DEFAULT_ARTIFACT_DATA_PATH
        if artifact_data_path_config is not None:
            if isinstance(artifact_data_path_config, list) and all(isinstance(item, str) for item in artifact_data_path_config):
                final_artifact_data_path = artifact_data_path_config
            else:
                self.logger.warning(f"Task '{task_name}', ConcatMethod: Invalid 'artifact_data_path' ('{artifact_data_path_config}'). Expected List[str]. Using default: {self.DEFAULT_ARTIFACT_DATA_PATH}.")

        # Validate and get artifact_data_key
        artifact_data_key = config.get(self.CONFIG_KEY_ARTIFACT_DATA_KEY)
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
        artifact_metrics_path_config = config.get(self.CONFIG_KEY_ARTIFACT_METRICS_PATH)
        final_artifact_metrics_path: List[str] = self.DEFAULT_ARTIFACT_DATA_PATH # Use the same default list type
        if artifact_metrics_path_config is not None:
            if isinstance(artifact_metrics_path_config, list) and all(isinstance(item, str) for item in artifact_metrics_path_config):
                final_artifact_metrics_path = artifact_metrics_path_config
            else:
                self.logger.warning(f"Task '{task_name}', CollectMetrics: Invalid 'artifact_metrics_path' ('{artifact_metrics_path_config}'). Expected List[str]. Using default: {self.DEFAULT_ARTIFACT_DATA_PATH}.")

        # Validate and get metric_definitions
        metric_definitions_config = config.get(self.CONFIG_KEY_METRIC_DEFINITIONS)
        if not isinstance(metric_definitions_config, list):
             self.logger.error(f"Task '{task_name}', CollectMetrics: 'metric_definitions' missing or invalid in config. Expected List.")
             return None
        final_metric_definitions: List[Dict[str, Any]] = metric_definitions_config # Assume list of dicts based on expected structure

        # Validate and get output_columns
        output_columns_config = config.get(self.CONFIG_KEY_OUTPUT_COLUMNS)
        final_output_columns: List[str] = self.DEFAULT_COLLECT_METRICS_OUTPUT_COLS
        if output_columns_config is not None:
            if isinstance(output_columns_config, list) and all(isinstance(item, str) and item.strip() for item in output_columns_config):
                final_output_columns = [col.strip() for col in output_columns_config]
            else:
                self.logger.warning(f"Task '{task_name}', CollectMetrics: Invalid 'output_columns' ('{output_columns_config}'). Expected List[str]. Using default: {self.DEFAULT_COLLECT_METRICS_OUTPUT_COLS}.")

        collected_rows: List[Dict[str, Any]] = []
        for p_idx, artifact in enumerate(all_participant_artifacts):
            p_id = artifact.get('participant_id', f"P{p_idx:03d}")
            metrics_location = artifact
            current_metric_definition_context = "N/A (before processing specific definitions)"  # For robust error logging
            try:
                for key_part in final_artifact_metrics_path:
                    if not isinstance(metrics_location, dict):
                        self.logger.warning(
                            f"Task '{task_name}', CollectMetrics: Path traversal for P:{p_id} encountered non-dict "
                            f"before key '{key_part}'. Path: {final_artifact_metrics_path}. Stopping traversal."
                        )
                        metrics_location = {}  # Ensure it's a dict to allow the next check to proceed
                        break
                    metrics_location = metrics_location.get(key_part, {})

                if not isinstance(metrics_location, dict):
                    self.logger.debug(f"Task '{task_name}', CollectMetrics: Metrics location at path {final_artifact_metrics_path} is not a dict for P:{p_id}. Skipping participant.")
                    continue

                for i, metric_def_item in enumerate(final_metric_definitions):
                    current_metric_definition_context = f"definition {i}: {str(metric_def_item)}"

                    if not isinstance(metric_def_item, dict):
                        self.logger.warning(f"Task '{task_name}', CollectMetrics: Metric definition {i} is not a dictionary. Value: {metric_def_item}. Skipping this definition.")
                        continue


                    metric_key = metric_def_item.get(self.METRIC_DEF_KEY_METRIC_KEY)
                    metric_key_template = metric_def_item.get(self.METRIC_DEF_KEY_METRIC_KEY_TEMPLATE)
                    output_metric_name = metric_def_item.get(self.METRIC_DEF_KEY_OUTPUT_METRIC_NAME) # Used in direct key branch

                    # Validate metric definition structure
                    is_direct_key = isinstance(metric_key, str) and metric_key.strip()
                    is_template_key = isinstance(metric_key_template, str) and metric_key_template.strip()

                    if not is_direct_key and not is_template_key:
                        self.logger.warning(f"Task '{task_name}', CollectMetrics: Metric definition {i} for P:{p_id} is missing both 'metric_key' and 'metric_key_template' or they are invalid. Definition: {metric_def_item}. Skipping this definition.")
                        continue # Skip this invalid definition

                    if is_direct_key: # Direct metric key
                        # metric_key is guaranteed to be a non-empty string here by is_direct_key
                        assert isinstance(metric_key, str), "metric_key should be a string at this point"
                        value = metrics_location.get(metric_key)
                        if value is not None:
                            row_data = {'participant_id': p_id, 'metric_name': output_metric_name or metric_key, 'value': value}
                            for col_name, col_value in metric_def_item.items(): # Add other fixed columns
                                if col_name not in [self.METRIC_DEF_KEY_METRIC_KEY, self.METRIC_DEF_KEY_OUTPUT_METRIC_NAME]:
                                    row_data[col_name] = col_value
                            collected_rows.append(row_data)
                    elif is_template_key: # Templated metric key
                        iterate_items_key = metric_def_item.get(self.METRIC_DEF_KEY_ITERATE_ITEMS_KEY)
                        iteration_list = self.general_configs.get(iterate_items_key, [None]) if iterate_items_key else [None]
                        if iterate_items_key and not self.general_configs.get(iterate_items_key):
                            self.logger.warning(f"Task '{task_name}', CollectMetrics: 'iterate_items_from_general_config_key' '{iterate_items_key}' not found or empty in general_configs.")

                        for item in iteration_list:
                            actual_key = metric_key_template
                            try:
                                if item is not None:
                                    # is_template_key ensures metric_key_template is a non-empty string.
                                    # Adding an explicit None check for extreme safety.
                                    if metric_key_template is not None:
                                        self.logger.debug(
                                            f"Task '{task_name}', CollectMetrics: Pre-format metric_key_template (with item). "
                                            f"Type: {type(metric_key_template)}, Value: '{metric_key_template}', Item: '{item}'"
                                        )
                                        item_str = str(item)
                                        actual_key = metric_key_template.format(item=item_str)

                                    else:
                                        # This path should ideally not be hit if is_template_key is true.
                                        self.logger.error(f"Task '{task_name}', CollectMetrics: metric_key_template is unexpectedly None before format (with item). PID: {p_id}, Def: {metric_def_item}. Using fallback key.")
                                        actual_key = f"fallback_key_template_none_item_{str(item)}" # Ensure actual_key is a string
                                else: # item is None
                                    try:
                                        # is_template_key ensures metric_key_template is a non-empty string.
                                        # Adding an explicit None check for extreme safety.
                                        if metric_key_template is not None:
                                            self.logger.debug(
                                                f"Task '{task_name}', CollectMetrics: Pre-format metric_key_template (no item). "
                                                f"Type: {type(metric_key_template)}, Value: '{metric_key_template}'")
                                            actual_key = metric_key_template.format()

                                        else:
                                            # This path should ideally not be hit.
                                            self.logger.error(f"Task '{task_name}', CollectMetrics: metric_key_template is unexpectedly None before format (no item). PID: {p_id}, Def: {metric_def_item}. Using fallback key.")
                                            actual_key = "fallback_key_template_none_no_item" # Ensure actual_key is a string
                                    except (KeyError, IndexError, Exception) as e_format_no_item:
                                        self.logger.warning(
                                            f"Task '{task_name}', CollectMetrics: Failed to format metric_key_template (when item is None). "
                                            f"Template at time of error: Type={type(metric_key_template)}, Value='{metric_key_template}'. "
                                            f"Error: {e_format_no_item}. "
                                            f"Using template string as key."
                                        )
                                        # actual_key remains metric_key_template (the initial fallback)
                            except Exception as e_format_with_item: # Catches errors from metric_key_template.format(item=item_str) or other unexpected errors.
                                self.logger.warning(
                                    f"Task '{task_name}', CollectMetrics: Error formatting metric_key_template (with item). "
                                    f"Template at time of error: Type={type(metric_key_template)}, Value='{metric_key_template}'. "
                                    f"Item='{item}'. Error: {e_format_with_item}. "
                                    f"Using template as key."
                                )
                                # actual_key remains metric_key_template (the initial fallback)

                            # Ensure actual_key is a string before using it as a dictionary key
                            if not isinstance(actual_key, str):
                                self.logger.error(
                                    f"Task '{task_name}', CollectMetrics: 'actual_key' is not a string before retrieving value. "
                                    f"Type: {type(actual_key)}, Value: '{actual_key}'. PID: {p_id}, Def: {metric_def_item}, Item: {item}. "
                                    f"Skipping this metric for this item."
                                )
                                continue # Skip to the next item in the iteration_list


                            value = metrics_location.get(actual_key)
                            if value is not None and isinstance(actual_key, str): # Ensure actual_key is a string before using it
                                row_data = {'participant_id': p_id, 'value': value}
                                out_metric_name_tpl = metric_def_item.get(self.METRIC_DEF_KEY_OUTPUT_METRIC_NAME_TEMPLATE)
                                if isinstance(out_metric_name_tpl, str) and out_metric_name_tpl.strip():
                                    try: # Ensure out_metric_name_tpl is a string before calling format
                                        # The outer 'if' should guarantee out_metric_name_tpl is a non-empty string.
                                        # Adding an explicit None check for extreme safety.
                                        if out_metric_name_tpl is None: # This should be impossible if outer 'if' is true.
                                            self.logger.error(f"Task '{task_name}', CollectMetrics: out_metric_name_tpl is unexpectedly None within guarded block. PID: {p_id}, Def: {metric_def_item}. Using actual_key as metric name.")
                                            row_data['metric_name'] = actual_key # Fallback
                                        elif item is not None:
                                            self.logger.debug(
                                                f"Task '{task_name}', CollectMetrics: Pre-format out_metric_name_tpl (with item). "
                                                f"Type: {type(out_metric_name_tpl)}, Value: '{out_metric_name_tpl}', Item: '{item}'"
                                            )
                                            item_str = str(item)
                                            row_data['metric_name'] = out_metric_name_tpl.format(item=item_str)
                                        else: # item is None
                                            # If template requires 'item', .format() (with no args) will raise KeyError, caught below.
                                            self.logger.debug(
                                                f"Task '{task_name}', CollectMetrics: Pre-format out_metric_name_tpl (no item). "
                                                f"Type: {type(out_metric_name_tpl)}, Value: '{out_metric_name_tpl}'"
                                            )
                                            row_data['metric_name'] = out_metric_name_tpl.format()

                                    except (KeyError, IndexError, Exception) as e_format_name: # Catch if format fails (incl. if out_metric_name_tpl was None and the above check was bypassed)
                                        item_str_for_log = str(item) if item is not None else "None"
                                        self.logger.warning(
                                            f"Task '{task_name}', CollectMetrics: Could not format output_metric_name_template. "
                                            f"Template at time of error: Type={type(out_metric_name_tpl)}, Value='{out_metric_name_tpl}'. "
                                            f"Item='{item_str_for_log}'. Error: {e_format_name}. "
                                            f"Using actual_key as metric name."
                                        )
                                        row_data['metric_name'] = actual_key # Fallback
                                else:
                                    row_data['metric_name'] = actual_key # Default to actual_key

                                # Validate and add output_iteration_column
                                output_iter_col_name = metric_def_item.get(self.METRIC_DEF_KEY_OUTPUT_ITERATION_COLUMN)
                                if isinstance(output_iter_col_name, str) and output_iter_col_name.strip() and item is not None:
                                    row_data[output_iter_col_name] = item
                                for col_name, col_value in metric_def_item.items(): # Add other fixed columns
                                    if col_name not in [self.METRIC_DEF_KEY_METRIC_KEY_TEMPLATE,
                                                        self.METRIC_DEF_KEY_ITERATE_ITEMS_KEY,
                                                        self.METRIC_DEF_KEY_OUTPUT_ITERATION_COLUMN,
                                                        self.METRIC_DEF_KEY_OUTPUT_METRIC_NAME_TEMPLATE]:
                                        row_data[col_name] = col_value # Use metric_def_item here as it's the source of the definition
                                collected_rows.append(row_data)
            except Exception as e:
                self.logger.error(f"Task '{task_name}', CollectMetrics: Error processing metrics for P:{p_id} (context: {current_metric_definition_context}): {e}", exc_info=True)

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