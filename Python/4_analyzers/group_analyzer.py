import os
import pandas as pd
import numpy as np
import json # type: ignore
from typing import List, Dict, Any, Optional, Union # For type hinting
import re # For potential use in parsing column names during aggregation

# Helper class for JSON encoding numpy types, if not already globally available
class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle pandas NaT (Not a Time)
        elif isinstance(obj, pd.NaT):
            return None # Or a specific string like "NaT"
        # Handle pandas NaN
        elif isinstance(obj, float) and np.isnan(obj):
            return None # Or a specific string like "NaN"
        else:
            return super(NpEncoder, self).default(obj)

class GroupAnalyzer:
    def __init__(self, main_logger, output_base_dir: str, 
                 analysis_tasks_config: List[Dict[str, Any]], 
                 available_analyzers: Dict[str, Any],
                 general_configs: Optional[Dict[str, Any]] = None
                 ):
        """
        Initializes the GroupAnalyzer.

        Args:
            main_logger: The main logger instance.
            output_base_dir (str): The base directory for all outputs.
            analysis_tasks_config (List[Dict[str, Any]]): A list of dictionaries,
                                                         each defining a data aggregation or analysis task.
            available_analyzers (Dict[str, Any]): A dictionary mapping string keys
                                                 to instantiated analyzer objects (e.g., ANOVAAnalyzer, CorrelationAnalyzer).
            general_configs (Optional[Dict[str, Any]]): Optional dictionary for general configurations
                                                        (e.g., lists of conditions, band definitions)
                                                        that might be needed by specific tasks.
        """
        self.logger = main_logger
        self.output_base_dir = output_base_dir
        self.group_results_dir = os.path.join(output_base_dir, "_GROUP_RESULTS")
        os.makedirs(self.group_results_dir, exist_ok=True)
        self.logger.info(f"Group results will be saved in: {self.group_results_dir}")

        # Store general configurations that might be used by various tasks
        # GroupAnalyzer itself doesn't assume specific keys like 'emotional_conditions'
        self.general_configs: Dict[str, Any] = general_configs if general_configs is not None else {}
        self.logger.info(f"GroupAnalyzer initialized with general_configs keys: {list(self.general_configs.keys())}")

        self.ANALYSIS_TASKS_CONFIG_LIST = analysis_tasks_config
        if not self.ANALYSIS_TASKS_CONFIG_LIST:
            self.logger.warning("GroupAnalyzer initialized with an empty 'analysis_tasks_config'. Specific analyses might not run or might use defaults.")
        
        self.available_analyzers = available_analyzers
        if not self.available_analyzers:
            self.logger.warning("GroupAnalyzer initialized with an empty 'available_analyzers' dictionary. Statistical operations will likely fail.")

        self.all_group_level_results: Dict[str, Any] = {} 
        self.aggregated_dataframes: Dict[str, pd.DataFrame] = {} # Stores DataFrames created by aggregation tasks

    def _prepare_data_for_task(self, task_config: Dict[str, Any], all_participant_artifacts: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        Prepares (aggregates, filters, reshapes) data for a given analysis task based on its configuration.
        This method handles both using previously aggregated data and performing new aggregations.

        Args:
            task_config (Dict[str, Any]): Configuration dictionary for the current task.
            all_participant_artifacts (List[Dict[str, Any]]): List of dictionaries, where each dict
                                                             contains the processed artifacts for one participant.

        Returns:
            Optional[pd.DataFrame]: The prepared DataFrame for the task, or None if preparation failed.
        """
        task_name = task_config.get('task_name', 'Unnamed Task')
        self.logger.info(f"Preparing data for task: {task_name}")
        
        data_source_df_name = task_config.get('data_source_df_name')
        aggregation_config = task_config.get('aggregation_config')
        
        current_df: Optional[pd.DataFrame] = None

        if data_source_df_name: # Use a previously aggregated DataFrame
            if data_source_df_name in self.aggregated_dataframes:
                current_df = self.aggregated_dataframes[data_source_df_name].copy()
                self.logger.info(f"Using previously aggregated DataFrame: '{data_source_df_name}' (shape: {current_df.shape})")
            else:
                self.logger.error(f"Data source DataFrame '{data_source_df_name}' not found in internal aggregated_dataframes for task '{task_name}'.")
                return None
        elif aggregation_config: # Perform new aggregation from participant artifacts
            method = aggregation_config.get('method')
            output_df_name = aggregation_config.get('output_df_name') # Name to store the aggregated df
            
            if not output_df_name:
                 self.logger.error(f"Task '{task_name}': 'output_df_name' is required in aggregation_config.")
                 return None

            if method == "concat_dataframes_from_artifacts":
                # Config needs: artifact_data_path (list), artifact_data_key (str)
                artifact_data_path = aggregation_config.get('artifact_data_path', [])
                artifact_data_key = aggregation_config.get('artifact_data_key')
                if not artifact_data_key:
                    self.logger.error(f"Task '{task_name}': 'artifact_data_key' missing in aggregation_config for method 'concat_dataframes_from_artifacts'.")
                    return None
                
                temp_dfs: List[pd.DataFrame] = []
                for p_idx, artifact in enumerate(all_participant_artifacts):
                    p_id = artifact.get('participant_id', f"P{p_idx:03d}")
                    data_location = artifact
                    try:
                        # Traverse the dict path to find the DataFrame
                        for key_part in artifact_data_path:
                            data_location = data_location.get(key_part, {})
                        df_to_add = data_location.get(artifact_data_key)

                        if isinstance(df_to_add, pd.DataFrame) and not df_to_add.empty:
                            df_to_add_copy = df_to_add.copy()
                            # Ensure participant_id is present
                            if 'participant_id' not in df_to_add_copy.columns:
                                df_to_add_copy['participant_id'] = p_id
                            temp_dfs.append(df_to_add_copy)
                        elif df_to_add is not None:
                             self.logger.debug(f"Data for key '{artifact_data_key}' in P:{p_id} is not a non-empty DataFrame (type: {type(df_to_add)}). Skipping.")
                    except AttributeError:
                        self.logger.debug(f"Path {artifact_data_path} not fully found or invalid in artifact for P:{p_id}, key {artifact_data_key}")
                
                if temp_dfs:
                    current_df = pd.concat(temp_dfs, ignore_index=True)
                    self.logger.info(f"Aggregated data using '{method}' for key '{artifact_data_key}'. Result shape: {current_df.shape}")
                else:
                    self.logger.warning(f"No data found to aggregate using '{method}' for key '{artifact_data_key}'.")
                    current_df = pd.DataFrame()
            
            elif method == "collect_metrics_to_dataframe":
                # Configuration for this method:
                # "artifact_metrics_path": ["path", "to", "metrics_dict_in_artifact"]
                # "metric_definitions": [
                #    {"metric_key": "actual_key_in_metrics_dict", "output_metric_name": "name_for_metric_type_col", "condition": "OptionalConditionName"},
                #    {"metric_key_template": "prefix_{condition}_suffix", "iterate_items_from_general_config_key": "my_study_conditions_key", 
                #     "output_metric_name_template": "prefix_suffix", "output_iteration_column": "condition"}
                # ]
                # "output_columns": ["participant_id", "condition", "metric_type", "value"] (example)
                
                artifact_metrics_path = aggregation_config.get('artifact_metrics_path', [])
                metric_definitions = aggregation_config.get('metric_definitions', [])
                output_columns = aggregation_config.get('output_columns', ['participant_id', 'metric_name', 'value']) # Default output structure

                collected_rows: List[Dict[str, Any]] = []
                for p_idx, artifact in enumerate(all_participant_artifacts):
                    p_id = artifact.get('participant_id', f"P{p_idx:03d}")
                    metrics_location = artifact
                    try:
                        # Traverse the dict path to find the metrics dictionary
                        for key_part in artifact_metrics_path:
                            metrics_location = metrics_location.get(key_part, {})
                        if not isinstance(metrics_location, dict):
                            self.logger.debug(f"Metrics location at path {artifact_metrics_path} is not a dict for P:{p_id}. Skipping.")
                            continue

                        for metric_def in metric_definitions:
                            metric_key = metric_def.get('metric_key')
                            metric_key_template = metric_def.get('metric_key_template')
                            output_metric_name = metric_def.get('output_metric_name') # For the 'metric_name' column
                            
                            if metric_key: # Direct metric key
                                value = metrics_location.get(metric_key)
                                if value is not None:
                                    row_data = {'participant_id': p_id, 'metric_name': output_metric_name or metric_key, 'value': value}
                                    # Allow adding other fixed columns from the definition
                                    for col_name, col_value in metric_def.items():
                                        if col_name not in ['metric_key', 'output_metric_name']:
                                             row_data[col_name] = col_value
                                    collected_rows.append(row_data)
                            elif metric_key_template: # Templated metric key (e.g., iterating over conditions, bands, etc.)
                                # Key in general_configs that holds the list of items to iterate over
                                iterate_items_key = metric_def.get('iterate_items_from_general_config_key') 
                                iteration_list: List[Any] = []
                                if iterate_items_key:
                                    iteration_list = self.general_configs.get(iterate_items_key, [])
                                    if not iteration_list:
                                        self.logger.warning(f"Task '{task_name}', method 'collect_metrics_to_dataframe': iteration_items_key '{iterate_items_key}' not found in general_configs or is empty.")
                                        continue # Skip this metric definition if iteration list is empty
                                else: # No iteration key provided, assume template is used as is or with a single placeholder if needed
                                    iteration_list = [None] # Iterate once

                                output_iteration_column = metric_def.get('output_iteration_column') # Column name for the iterated item (e.g., 'condition', 'band')
                                output_metric_name_template = metric_def.get('output_metric_name_template') # Template for the output metric_name column value

                                for item in iteration_list:
                                    # Format the actual metric key using the item
                                    try:
                                        actual_key = metric_key_template.format(item=item) if item is not None else metric_key_template.format() # Handle template with/without placeholder
                                    except (KeyError, IndexError): # Handle cases where template expects {item} but item is None
                                         if item is None:
                                             actual_key = metric_key_template # Use template as is if item is None
                                         else:
                                             self.logger.error(f"Task '{task_name}', metric_def: Could not format metric_key_template '{metric_key_template}' with item '{item}'. Skipping.")
                                             continue

                                    value = metrics_location.get(actual_key)
                                    if value is not None:
                                        row_data = {'participant_id': p_id, 'value': value}
                                        
                                        # Determine the value for the 'metric_name' column
                                        if output_metric_name_template:
                                            try:
                                                row_data['metric_name'] = output_metric_name_template.format(item=item) if item is not None else output_metric_name_template.format()
                                            except (KeyError, IndexError):
                                                 self.logger.warning(f"Task '{task_name}', metric_def: Could not format output_metric_name_template '{output_metric_name_template}' with item '{item}'. Using actual key.")
                                                 row_data['metric_name'] = actual_key
                                        else:
                                            row_data['metric_name'] = actual_key # Default to actual key if no template

                                        # Add the iterated item to its specified column
                                        if output_iteration_column and item is not None:
                                            row_data[output_iteration_column] = item
                                            
                                        # Allow adding other fixed columns from the definition
                                        for col_name, col_value in metric_def.items():
                                            if col_name not in ['metric_key_template', 'iterate_items_from_general_config_key', 
                                                                'output_iteration_column', 'output_metric_name_template']:
                                                row_data[col_name] = col_value

                                        collected_rows.append(row_data)

                    except AttributeError:
                        self.logger.debug(f"Path {artifact_metrics_path} not fully found or invalid in artifact for P:{p_id} for metric collection.")
                    except Exception as e_metric_def:
                         self.logger.error(f"Error processing metric definition {metric_def} for P:{p_id} in task '{task_name}': {e_metric_def}", exc_info=True)

                if collected_rows:
                    current_df = pd.DataFrame(collected_rows)
                    # Ensure all defined output_columns exist, fill with NaN if not produced by collection
                    # and enforce the specified order.
                    for col in output_columns:
                        if col not in current_df.columns:
                            current_df[col] = np.nan

                    current_df = current_df[output_columns] # Select and order according to output_columns
                    self.logger.info(f"Collected metrics into DataFrame. Shape: {current_df.shape}")
                else:
                    self.logger.warning(f"No metrics collected for task '{task_name}'.")
                    current_df = pd.DataFrame(columns=output_columns) # Return empty DF with expected columns


            # TODO: Implement other aggregation methods like 'merge_dataframes_from_artifacts', 
            #       'pivot_table_from_artifact_data', 'group_and_aggregate_within_artifacts'
            #       Each method would interpret its specific sub-configuration within aggregation_config.
            #       Example: 'merge_dataframes_from_artifacts' config:
            #       { "method": "merge_dataframes_from_artifacts",
            #         "output_df_name": "merged_trial_data",
            #         "merge_definitions": [
            #            {"artifact_data_path": ["path", "to", "df1"], "artifact_data_key": "df1_key"},
            #            {"artifact_data_path": ["path", "to", "df2"], "artifact_data_key": "df2_key"}
            #         ],
            #         "on": "common_column",
            #         "how": "inner"
            #       }
            #       Example: 'group_and_aggregate_within_artifacts' config:
            #       { "method": "group_and_aggregate_within_artifacts",
            #         "output_df_name": "participant_avg_per_condition",
            #         "source_artifact_data_path": ["path", "to", "source_df"],
            #         "source_artifact_data_key": "source_df_key",
            #         "group_by": ["condition", "eeg_band"],
            #         "agg_dict": {"plv": "mean", "amplitude": "mean"}
            #       }

            else:
                self.logger.error(f"Unknown aggregation method: '{method}' for task '{task_name}'.")
                return None

            # Store the newly aggregated DataFrame
            if output_df_name and current_df is not None: # current_df could be empty but not None
                self.aggregated_dataframes[output_df_name] = current_df.copy()
                self.logger.info(f"Stored aggregated DataFrame as: '{output_df_name}'")
                if aggregation_config.get('save_aggregated_df', False):
                    save_filename = aggregation_config.get('save_filename_template', f"aggregated_{output_df_name}.csv")
                    save_path = os.path.join(self.group_results_dir, save_filename)
                    try:
                        current_df.to_csv(save_path, index=False)
                        self.logger.info(f"Saved aggregated DataFrame to: {save_path}")
                    except Exception as e_save_agg:
                        self.logger.error(f"Failed to save aggregated DataFrame {save_path}: {e_save_agg}")
            
        else: # Neither data_source_df_name nor aggregation_config provided
            self.logger.error(f"Task '{task_name}' must specify either 'data_source_df_name' or 'aggregation_config'.")
            return None

        # Apply filters if specified (applied after getting the initial DF, whether from source or aggregation)
        filter_conditions = task_config.get('data_filter_conditions')
        if current_df is not None and not current_df.empty and filter_conditions and isinstance(filter_conditions, dict):
            self.logger.info(f"Applying data filters: {filter_conditions} to DataFrame for task '{task_name}'.")
            original_rows = len(current_df)
            try:
                for col, value in filter_conditions.items():
                    if col in current_df.columns:
                        if isinstance(value, list): 
                            current_df = current_df[current_df[col].isin(value)]
                        else: 
                            current_df = current_df[current_df[col] == value]
                    else:
                        self.logger.warning(f"Filter column '{col}' not found in DataFrame for task '{task_name}'. Skipping filter for this column.")
                self.logger.info(f"DataFrame rows after filtering: {len(current_df)} (was {original_rows}).")
                if current_df.empty:
                    self.logger.warning(f"DataFrame became empty after applying filters for task '{task_name}'.")
                    # Return empty DF, let analysis step decide if it can proceed
            except Exception as e_filter:
                 self.logger.error(f"Error applying filters for task '{task_name}': {e_filter}", exc_info=True)
                 # Decide how to handle filter errors - returning None might be too harsh
                 # Returning the pre-filtered DF or an empty DF might be better
                 return pd.DataFrame() # Return empty DF on filter error

        if current_df is None: # Should only happen if initial selection/aggregation failed
             self.logger.warning(f"No data prepared for task '{task_name}'.")
        # Allow empty current_df to be returned, analysis step will handle it.
        
        return current_df

    def _execute_analysis_step(self, task_config: Dict[str, Any], data_df: pd.DataFrame) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        """Executes the statistical analysis step of a task."""
        analysis_config = task_config.get('analysis_step')
        task_name = task_config.get('task_name', 'Unnamed Analysis Task')

        if not analysis_config:
            self.logger.debug(f"No analysis step defined for task: {task_name}")
            return None
        if data_df is None: # Check for None specifically, empty DF might be valid for some checks
            self.logger.warning(f"Input data_df is None for analysis step in task: {task_name}. Skipping analysis.")
            return None
        if data_df.empty: # If there's an analysis step but data is empty, warn.
            self.logger.warning(f"Input data_df is empty for analysis step in task: {task_name}. Analysis might fail or produce no results.")


        analyzer_key = analysis_config.get('analyzer_key')
        method_to_call_str = analysis_config.get('method_to_call')
        method_params = analysis_config.get('method_params', {}).copy() 

        analyzer_instance = self.available_analyzers.get(analyzer_key)
        if not analyzer_instance:
            self.logger.error(f"Analyzer '{analyzer_key}' not found in available_analyzers for task '{task_name}'.")
            return None

        if not hasattr(analyzer_instance, method_to_call_str):
            self.logger.error(f"Method '{method_to_call_str}' not found in analyzer '{analyzer_key}' for task '{task_name}'.")
            return None

        method_to_call = getattr(analyzer_instance, method_to_call_str)
        
        try:
            self.logger.info(f"Executing analysis: {analyzer_key}.{method_to_call_str} for task '{task_name}'.")
            
            # --- Flexible Argument Passing ---
            # Prioritize explicit data argument name from config
            data_arg_name_config = analysis_config.get("data_argument_name")
            
            sig = method_to_call.__code__
            arg_names = sig.co_varnames[:sig.co_argcount]

            analysis_results: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None

            if data_arg_name_config and data_arg_name_config in arg_names:
                # Use the configured argument name
                kwargs = {data_arg_name_config: data_df, **method_params}
                analysis_results = method_to_call(**kwargs)
            elif "data_df" in arg_names: 
                # Fallback to common 'data_df' argument name
                analysis_results = method_to_call(data_df=data_df, **method_params)
            elif "series1" in arg_names and "series1_col" in method_params and "series2_col" in method_params:
                # Handle correlation-like methods expecting series
                series1_col = method_params.pop("series1_col")
                series2_col = method_params.pop("series2_col")
                if series1_col not in data_df.columns or series2_col not in data_df.columns:
                    self.logger.error(f"Missing series columns '{series1_col}' or '{series2_col}' in data_df for correlation task '{task_name}'.")
                    analysis_results = None # Set to None, error already logged.
                else:
                    analysis_results = method_to_call(series1=data_df[series1_col], series2=data_df[series2_col], **method_params)
            else: 
                # Last resort: call with only method_params. Assumes method doesn't need a direct DF arg
                # or gets data via other means specified in method_params.
                self.logger.debug(f"Calling {method_to_call_str} for task '{task_name}' with only keyword arguments from method_params. "
                                  "Ensure the method does not require a direct DataFrame/series if not matching 'data_df', 'series1/2', or configured 'data_argument_name'.")
                analysis_results = method_to_call(**method_params)
                
            self.logger.info(f"Analysis method '{method_to_call_str}' completed for task '{task_name}'.")
            return analysis_results
        except Exception as e_analysis:
            self.logger.error(f"Error during analysis step for task '{task_name}': {e_analysis}", exc_info=True)
            return None

    def run_group_analysis(self, all_participant_artifacts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Main orchestrator for group-level analyses. Iterates through configured tasks,
        performs data preparation/aggregation, and executes statistical analyses.

        Args:
            all_participant_artifacts (List[Dict[str, Any]]): A list of dictionaries, where each dict
                                                             contains the processed artifacts for one participant.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are task names and values are dictionaries
                                       containing the 'prepared_data' (DataFrame) and 'analysis_results'
                                       (DataFrame or dict) for each task. Includes error info if a task failed.
        """
        if not all_participant_artifacts:
            self.logger.warning("No participant artifacts provided for group analysis. Skipping.")
            return {}
        
        task_outputs: Dict[str, Dict[str, Any]] = {}
        # Reset aggregated_dataframes for a new run
        self.aggregated_dataframes = {}

        self.logger.info(f"--- Starting Group-Level Analysis with {len(all_participant_artifacts)} participant artifacts ---")

        for task_config in self.ANALYSIS_TASKS_CONFIG_LIST:
            task_name = task_config.get('task_name', f"Unnamed_Task_{len(task_outputs) + 1}")
            if not task_config.get('run_task', False):
                self.logger.info(f"Skipping task '{task_name}' as 'run_task' is false or not set.")
                continue

            self.logger.info(f"--- Group Analysis: Processing Task '{task_name}' ---")
            task_type = task_config.get('task_type') 
            
            try:
                # Step 1: Prepare/Aggregate data for the task
                prepared_df = self._prepare_data_for_task(task_config, all_participant_artifacts)

                current_task_output: Dict[str, Any] = {"prepared_data": prepared_df, "analysis_results": None}

                # If the task is purely for data aggregation (and doesn't have an analysis step)
                if task_type == "data_aggregation" and not task_config.get("analysis_step"):
                    self.logger.info(f"Data aggregation task '{task_name}' completed.")
                    # The aggregated DF is stored in self.aggregated_dataframes by _prepare_data_for_task
                    task_outputs[task_name] = current_task_output
                    continue 
                
                # If prepared_df is None (error during prep)
                if prepared_df is None:
                    self.logger.warning(f"No data prepared for task '{task_name}'. Skipping further steps for this task.")
                    task_outputs[task_name] = current_task_output
                    continue
                
                # If prepared_df is empty and an analysis step is expected, warn.
                if prepared_df.empty and task_config.get("analysis_step"):
                     self.logger.warning(f"Prepared data is empty for task '{task_name}' which has an analysis step. Analysis may not run or produce meaningful results.")
                

                # Step 2: Execute statistical analysis if defined
                analysis_results_data: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None
                if task_config.get("analysis_step"):
                    analysis_results_data = self._execute_analysis_step(task_config, prepared_df)
                    if analysis_results_data is not None:
                        results_key = task_config.get('results_storage_key', f"{task_name}_results")
                        output_prefix = task_config.get('output_file_prefix', task_name.replace(" ", "_").replace("/", "_"))
                        
                        # Store raw result in the overall results dictionary
                        self.all_group_level_results[results_key] = analysis_results_data 
                        
                        # Save results to file based on type
                        if isinstance(analysis_results_data, pd.DataFrame):
                            try:
                                analysis_results_data.to_csv(os.path.join(self.group_results_dir, f"{output_prefix}_stats.csv"), index=False)
                                self.logger.info(f"Saved analysis results (DataFrame) for task '{task_name}' to CSV.")
                            except Exception as e_csv_save:
                                self.logger.error(f"Could not save DataFrame results for {task_name} as CSV: {e_csv_save}", exc_info=True)

                        elif isinstance(analysis_results_data, dict):
                            try:
                                with open(os.path.join(self.group_results_dir, f"{output_prefix}_stats.json"), 'w') as f_json:
                                    json.dump(analysis_results_data, f_json, indent=4, cls=NpEncoder)
                                self.logger.info(f"Saved analysis results (dict) for task '{task_name}' to JSON.")
                            except Exception as e_json_save:
                                self.logger.error(f"Could not save dict results for {task_name} as JSON: {e_json_save}", exc_info=True)
                        else: 
                             self.logger.warning(f"Analysis results for task '{task_name}' are of unhandled type {type(analysis_results_data)}. Not saving to dedicated file.")
                             # The result is still stored in self.all_group_level_results and will be in the summary JSON

                        current_task_output["analysis_results"] = analysis_results_data # Store results in output dict
                    else:
                         self.logger.warning(f"Analysis step for task '{task_name}' returned None.")
                
                task_outputs[task_name] = current_task_output # Store the output for this task

            except Exception as e_task: # Catch any unexpected errors during task processing
                self.logger.error(f"Error processing task '{task_name}': {e_task}", exc_info=True)
                self.all_group_level_results[f"{task_name}_error"] = str(e_task)
                # Store error info in task_outputs as well
                task_outputs[task_name] = {"prepared_data": None, "analysis_results": None, "error": str(e_task)}


        # Save all collected group-level statistical results to a JSON summary file
        summary_json_path = os.path.join(self.group_results_dir, "group_analysis_summary_all_stats.json")
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(self.all_group_level_results, f, indent=4, cls=NpEncoder)
            self.logger.info(f"All group analysis statistical outputs summarized in: {summary_json_path}")
        except Exception as e_save:
            self.logger.error(f"Failed to save all group analysis results summary: {e_save}", exc_info=True)

        self.logger.info("--- Group-Level Analysis Completed ---")
        return task_outputs