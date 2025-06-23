import pandas as pd
from typing import Dict, Any, Optional, Union

class GroupLevelAnalyzing:
    """
    Executes statistical analyses on prepared group-level data using specialized analyzer modules.
    """
    # Default parameters
    DEFAULT_ANALYSIS_TASK_NAME = "Unnamed Analysis Task"

    def __init__(self, logger, available_analyzers: Dict[str, Any]):
        self.logger = logger
        self.available_analyzers = available_analyzers
        if not self.available_analyzers:
            self.logger.warning("GroupLevelAnalyzing initialized with no 'available_analyzers'. Statistical operations will fail.")
        self.logger.info("GroupLevelAnalyzing initialized.")

    def analyze_data(self,
                     data_df: pd.DataFrame,
                     analysis_config: Dict[str, Any],
                     task_name: str = DEFAULT_ANALYSIS_TASK_NAME) -> Optional[pd.DataFrame]:
        """
        Executes a statistical analysis step based on the provided configuration.

        Args:
            data_df: The prepared (and possibly processed) group-level DataFrame.
            analysis_config: Configuration for the analysis step, including analyzer key,
                             method to call, and parameters.
            task_name: Name of the task for logging.

        Returns:
            The results of the analysis (a pandas DataFrame), or None on failure.
        """
        if data_df is None:
            self.logger.warning(f"Task '{task_name}': Input data_df is None. Skipping analysis.")
            return None
        if data_df.empty and analysis_config: # If analysis is expected but data is empty
            self.logger.warning(f"Task '{task_name}': Input data_df is empty. Analysis might fail or produce no results.")

        analyzer_key = analysis_config.get('analyzer_key') # Orchestrator provides these keys
        method_to_call_str = analysis_config.get('method_to_call')
        method_params_raw = analysis_config.get('method_params')
        method_params = method_params_raw.copy() if isinstance(method_params_raw, dict) else {}
        
        if not isinstance(analyzer_key, str) or not analyzer_key.strip():
            self.logger.error(f"Task '{task_name}': 'analyzer_key' is missing, not a string, or empty in analysis_config.")
            return None

        analyzer_instance = self.available_analyzers.get(analyzer_key) # analyzer_key is now guaranteed to be a str
        if analyzer_instance is None: # Check if the key existed and an instance was found
            self.logger.error(f"Task '{task_name}': Analyzer '{analyzer_key}' not found in available_analyzers.")
            return None

        if not isinstance(method_to_call_str, str) or not method_to_call_str.strip():
            self.logger.error(f"Task '{task_name}': 'method_to_call' is missing, not a string, or empty in analysis_config for analyzer '{analyzer_key}'.")
            return None

        if not hasattr(analyzer_instance, method_to_call_str):
            self.logger.error(f"Task '{task_name}': Method '{method_to_call_str}' not found in analyzer '{analyzer_key}'.")
            return None

        method_to_call = getattr(analyzer_instance, method_to_call_str)
        
        if not callable(method_to_call):
            self.logger.error(f"Task '{task_name}': Attribute '{method_to_call_str}' in analyzer '{analyzer_key}' is not a callable method.")
            return None


        try:
            self.logger.info(f"Task '{task_name}': Executing {analyzer_key}.{method_to_call_str}.")
            # Log the parameters that will be passed, for easier debugging
            final_kwargs_to_log = method_params.copy() # Start with a copy of user-defined params

            data_arg_name_from_config = analysis_config.get('data_argument_name')
            sig = method_to_call.__code__
            arg_names = sig.co_varnames[:sig.co_argcount]

            # method_params is already a safe copy for this specific call
            if data_arg_name_from_config and data_arg_name_from_config in arg_names:
                final_kwargs_to_log[data_arg_name_from_config] = "<DataFrame>" # Placeholder for logging
                method_params[data_arg_name_from_config] = data_df # Ensure passed data_df takes precedence
                self.logger.debug(f"Task '{task_name}': Calling with params: {final_kwargs_to_log}")
                return method_to_call(**method_params) # type: ignore[return-value]
            elif "data_df" in arg_names:
                final_kwargs_to_log["data_df"] = "<DataFrame>"
                method_params["data_df"] = data_df # Ensure passed data_df takes precedence
                self.logger.debug(f"Task '{task_name}': Calling with params: {final_kwargs_to_log}")
                return method_to_call(**method_params) # type: ignore[return-value]
            elif "series1" in arg_names and method_params.get("series1_col") and method_params.get("series2_col"):
                # .pop() modifies method_params in place, which is fine as it's a per-call copy
                s1_col = method_params.pop("series1_col")
                s2_col = method_params.pop("series2_col")
                if s1_col not in data_df.columns or s2_col not in data_df.columns:
                    self.logger.error(f"Task '{task_name}': Missing series columns '{s1_col}' or '{s2_col}'.")
                    return None
                self.logger.debug(f"Task '{task_name}': Calling with series1='{s1_col}', series2='{s2_col}', and other params: {method_params}")
                # Pass the remaining method_params after popping
                return method_to_call(series1=data_df[s1_col], series2=data_df[s2_col], **method_params) # type: ignore[return-value]
            else:
                self.logger.debug(f"Task '{task_name}': Calling {analyzer_key}.{method_to_call_str} with only keyword arguments from method_params. "
                                  f"Params: {final_kwargs_to_log}. Ensure the method does not require a direct DataFrame/series if not matching 'data_df', 'series1/2', or configured 'data_argument_name'.")
                return method_to_call(**method_params) # type: ignore[return-value]
        except Exception as e_analysis:
            self.logger.error(f"Task '{task_name}': Error during analysis: {e_analysis}", exc_info=True)
            return None