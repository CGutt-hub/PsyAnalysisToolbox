import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional

class ScoreAnalyzer:
    # Default parameters for score analysis
    DEFAULT_SAVE_RESULTS_FLAG = True
    DEFAULT_RESULTS_PREFIX = "scale_analysis"
    DEFAULT_CORRELATION_METHOD = "pearson"

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ScaleScoreAnalyzer initialized.")

    def analyze_scores(self,
                       scores_df: pd.DataFrame,
                       participant_id_col: str,
                       scale_columns: Optional[List[str]] = None,
                       output_dir: Optional[str] = None,
                       save_results: bool = DEFAULT_SAVE_RESULTS_FLAG,
                       results_prefix: str = DEFAULT_RESULTS_PREFIX
                       ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Analyzes questionnaire scale scores to compute descriptive statistics and correlations.

        Args:
            scores_df (pd.DataFrame): DataFrame containing participant IDs and scale scores.
            participant_id_col (str): Name of the column containing participant IDs.
            scale_columns (Optional[List[str]]): List of column names representing the scales to analyze.
                                                 If None, all numeric columns except participant_id_col
                                                 will be considered as scales.
            output_dir (Optional[str]): Directory to save the analysis results. Required if save_results is True.
            save_results (bool): If True, saves descriptive statistics and correlation matrix to CSV. Defaults to ScoreAnalyzer.DEFAULT_SAVE_RESULTS_FLAG.
            results_prefix (str): Prefix for the saved output files.

        Returns:
            Dict[str, Optional[pd.DataFrame]]: A dictionary containing DataFrames for:
                'descriptives': Descriptive statistics for each scale.
                'correlations': Correlation matrix between scales.
                Returns None for a key if the analysis could not be performed.
        """
        analysis_results: Dict[str, Optional[pd.DataFrame]] = {
            'descriptives': None,
            'correlations': None
        }

        if scores_df is None or scores_df.empty:
            self.logger.warning("ScaleScoreAnalyzer - Input scores_df is empty or None. Skipping analysis.")
            return analysis_results

        if not isinstance(participant_id_col, str) or not participant_id_col.strip():
            self.logger.error("ScaleScoreAnalyzer - participant_id_col must be a non-empty string. Skipping analysis.")
            return analysis_results
        if participant_id_col not in scores_df.columns:
            self.logger.error(f"ScaleScoreAnalyzer - Participant ID column '{participant_id_col}' not found in scores_df. Skipping analysis.")
            return analysis_results

        if scale_columns is not None and (not isinstance(scale_columns, list) or not all(isinstance(col, str) for col in scale_columns)):
            self.logger.warning("ScaleScoreAnalyzer - 'scale_columns' provided but is not a list of strings. Will attempt to infer columns.")
            scale_columns = None # Fallback to inference

        if save_results:
            if not output_dir or not isinstance(output_dir, str) or not output_dir.strip():
                self.logger.error("ScaleScoreAnalyzer - 'output_dir' must be a non-empty string if save_results is True. Disabling save.")
                save_results = False
            if not isinstance(results_prefix, str) or not results_prefix.strip():
                self.logger.error("ScaleScoreAnalyzer - 'results_prefix' must be a non-empty string if save_results is True. Using default 'scale_analysis'.")
                results_prefix = "scale_analysis" # Fallback to a generic prefix if invalid and saving


        # Identify scale columns for analysis
        actual_scale_columns: List[str] = [] # Initialize to ensure it's always bound

        if scale_columns:
            actual_scale_columns = [col for col in scale_columns if col in scores_df.columns and col != participant_id_col]
            if not actual_scale_columns:
                self.logger.warning("ScaleScoreAnalyzer - None of the specified scale_columns found in scores_df or all were PID. Attempting to infer.")
                scale_columns = None # Fallback to inference
            else:
                self.logger.info(f"ScaleScoreAnalyzer - Using specified scale columns: {actual_scale_columns}")
        
        if not scale_columns: # Infer if not provided or if specified ones were not found
            self.logger.info("ScaleScoreAnalyzer - Inferring scale columns (numeric columns excluding PID).")
            actual_scale_columns = scores_df.select_dtypes(include=np.number).columns.tolist()
            if participant_id_col in actual_scale_columns:
                actual_scale_columns.remove(participant_id_col)
            
            if not actual_scale_columns:
                self.logger.error("ScaleScoreAnalyzer - No numeric scale columns found to analyze. Skipping analysis.")
                return analysis_results
            self.logger.info(f"ScaleScoreAnalyzer - Inferred scale columns for analysis: {actual_scale_columns}")

        scales_data_df = scores_df[actual_scale_columns]

        # 1. Descriptive Statistics
        try:
            self.logger.info("ScaleScoreAnalyzer - Calculating descriptive statistics.")
            descriptives_df = scales_data_df.describe().transpose()
            analysis_results['descriptives'] = descriptives_df
            self.logger.debug(f"Descriptive statistics:\n{descriptives_df}")

            if save_results and output_dir:
                os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
                desc_path = os.path.join(output_dir, f"{results_prefix}_descriptive_stats.csv")
                descriptives_df.to_csv(desc_path)
                self.logger.info(f"ScaleScoreAnalyzer - Descriptive statistics saved to: {desc_path}")
        except Exception as e:
            self.logger.error(f"ScaleScoreAnalyzer - Error calculating descriptive statistics: {e}", exc_info=True)

        # 2. Correlation Matrix
        if len(actual_scale_columns) >= 2: # Need at least 2 scales for correlation
            try:
                self.logger.info("ScaleScoreAnalyzer - Calculating correlation matrix.")
                correlation_df = scales_data_df.corr(method=self.DEFAULT_CORRELATION_METHOD)
                analysis_results['correlations'] = correlation_df
                self.logger.debug(f"Correlation matrix:\n{correlation_df}")

                if save_results and output_dir:
                    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
                    corr_path = os.path.join(output_dir, f"{results_prefix}_correlation_matrix.csv")
                    correlation_df.to_csv(corr_path)
                    self.logger.info(f"ScaleScoreAnalyzer - Correlation matrix saved to: {corr_path}")
            except Exception as e:
                self.logger.error(f"ScaleScoreAnalyzer - Error calculating correlation matrix: {e}", exc_info=True)
        else:
            self.logger.info("ScaleScoreAnalyzer - Fewer than 2 scale columns available. Skipping correlation analysis.")

        self.logger.info("ScaleScoreAnalyzer - Analysis of scale scores completed.")
        return analysis_results