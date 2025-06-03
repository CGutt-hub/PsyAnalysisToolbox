import os
import numpy as np
import pandas as pd
import neurokit2 as nk

class HRVAnalyzer:
    # Default parameters for HRV analysis
    DEFAULT_NNI_COLUMN_NAME = 'NN_Interval_ms'
    DEFAULT_HRV_METRIC_KEY_TEMPLATE = "hrv_{metric_name}_{scope}" # e.g., hrv_rmssd_Overall

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("HRVAnalyzer initialized.")

    def calculate_rmssd_from_nni_array(self, nn_intervals_ms):
        """Calculates RMSSD directly from an array of NN intervals in milliseconds."""
        # This method is quite specific, so its internal logic doesn't lend itself to many external defaults.
        if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
            self.logger.warning("HRVAnalyzer (array) - Not enough NN intervals for RMSSD.")
            return np.nan
        try:
            diff_nn = np.diff(nn_intervals_ms)
            rmssd = np.sqrt(np.mean(diff_nn ** 2))
            self.logger.info(f"HRVAnalyzer (array) - Calculated RMSSD: {rmssd:.2f} ms")
            return rmssd
        except Exception as e:
            self.logger.error(f"HRVAnalyzer (array) - Error calculating RMSSD: {e}", exc_info=True)
            return np.nan

    def calculate_hrv_metrics_from_nni_file(self, 
                                            nn_intervals_path, 
                                            nni_column_name=DEFAULT_NNI_COLUMN_NAME):
        """
        Calculates overall HRV metrics (e.g., RMSSD) from NN intervals file.
        Args:
            nn_intervals_path (str): Path to the CSV file containing NN intervals.
            nni_column_name (str): Name of the column containing NN intervals in milliseconds. Defaults to HRVAnalyzer.DEFAULT_NNI_COLUMN_NAME.
        Returns:
            a dictionary of metrics.
        """
        hrv_metrics = {}
        rmssd_overall_key = self.DEFAULT_HRV_METRIC_KEY_TEMPLATE.format(metric_name="rmssd", scope="Overall")

        if nn_intervals_path is None or not os.path.exists(nn_intervals_path):
            self.logger.warning("HRVAnalyzer - NN intervals file not found. Skipping overall HRV calculation.")
            hrv_metrics[rmssd_overall_key] = np.nan # Ensure key exists even on failure
            return hrv_metrics

        self.logger.info(f"HRVAnalyzer - Calculating overall HRV metrics from file: {nn_intervals_path} using column: {nni_column_name}")
        try:
            nn_intervals_df = pd.read_csv(nn_intervals_path)
            
            if nni_column_name not in nn_intervals_df.columns:
                self.logger.error(f"HRVAnalyzer - Column '{nni_column_name}' not found in NNI file: {nn_intervals_path}")
                hrv_metrics[rmssd_overall_key] = np.nan
                return hrv_metrics
            
            nn_intervals_ms = nn_intervals_df[nni_column_name].dropna().values

            if len(nn_intervals_ms) < 2:
                 self.logger.warning("HRVAnalyzer - Not enough valid NN intervals to calculate overall HRV. Skipping.")
                 hrv_metrics[rmssd_overall_key] = np.nan
                 return hrv_metrics
            
            hrv_metrics[rmssd_overall_key] = self.calculate_rmssd_from_nni_array(nn_intervals_ms)
            self.logger.info("HRVAnalyzer - Overall HRV calculation from file completed.")
            return hrv_metrics
        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error calculating overall HRV from file: {e}", exc_info=True)
            hrv_metrics[rmssd_overall_key] = np.nan
            return hrv_metrics