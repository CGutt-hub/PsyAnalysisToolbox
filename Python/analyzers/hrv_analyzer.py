import os
import numpy as np
import pandas as pd
import mne # For MNE Epochs type hint
import neurokit2 as nk # For HRV calculations
from typing import Dict, List, Optional # Added for type hinting
class HRVAnalyzer:
    # Default parameters for HRV analysis
    DEFAULT_NNI_COLUMN_NAME = 'NN_Interval_ms'
    DEFAULT_HRV_METRIC_KEY_TEMPLATE = "hrv_{metric_name}_{scope}" # e.g., hrv_rmssd_Overall
    DEFAULT_EPOCHED_HRV_METRICS = ['RMSSD', 'SDNN', 'pNN50', 'LF', 'HF', 'LFHF']
 
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("HRVAnalyzer initialized.")

    def calculate_rmssd_from_nni_array(self, nn_intervals_ms: np.ndarray) -> float:
        """Calculates RMSSD directly from an array of NN intervals in milliseconds."""
        # This method is quite specific, so its internal logic doesn't lend itself to many external defaults.
        if not isinstance(nn_intervals_ms, np.ndarray):
            self.logger.error("HRVAnalyzer (array) - Input nn_intervals_ms must be a numpy array.")
            return np.nan
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
                                            nn_intervals_path: str, 
                                            nni_column_name: str = DEFAULT_NNI_COLUMN_NAME
                                            ) -> Dict[str, float]:
        """
        Calculates overall HRV metrics (e.g., RMSSD) from NN intervals file.
        Args:
            nn_intervals_path (str): Path to the CSV file containing NN intervals.
            nni_column_name (str): Name of the column containing NN intervals in milliseconds. Defaults to HRVAnalyzer.DEFAULT_NNI_COLUMN_NAME.
        Returns:
            Dict[str, float]: A dictionary of metrics (currently only RMSSD).
        """
        hrv_metrics: Dict[str, float] = {} # Initialize with type hint
        rmssd_overall_key = self.DEFAULT_HRV_METRIC_KEY_TEMPLATE.format(metric_name="rmssd", scope="Overall")
        hrv_metrics[rmssd_overall_key] = np.nan # Ensure key exists even on failure

        if not isinstance(nn_intervals_path, str) or not nn_intervals_path.strip():
            self.logger.error("HRVAnalyzer - nn_intervals_path must be a non-empty string.")
            return hrv_metrics

        if nn_intervals_path is None or not os.path.exists(nn_intervals_path):
            self.logger.warning("HRVAnalyzer - NN intervals file not found. Skipping overall HRV calculation.")
            return hrv_metrics

        self.logger.info(f"HRVAnalyzer - Calculating overall HRV metrics from file: {nn_intervals_path} using column: {nni_column_name}")
        try:
            if not isinstance(nni_column_name, str) or not nni_column_name.strip():
                self.logger.error("HRVAnalyzer - nni_column_name must be a non-empty string.")
                return hrv_metrics
            nn_intervals_df = pd.read_csv(nn_intervals_path)
            
            if nni_column_name not in nn_intervals_df.columns:
                self.logger.error(f"HRVAnalyzer - Column '{nni_column_name}' not found in NNI file: {nn_intervals_path}")
                hrv_metrics[rmssd_overall_key] = np.nan
                return hrv_metrics
            
            nn_intervals_ms = np.asarray(nn_intervals_df[nni_column_name].dropna().values)

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

    def calculate_hrv_for_epochs(self,
                                 epochs: mne.Epochs,
                                 rpeaks_samples: np.ndarray,
                                 original_sfreq: float,
                                 metrics_to_calculate: Optional[List[str]] = None,
                                 participant_id: Optional[str] = None
                                 ) -> Optional[pd.DataFrame]:
        """
        Calculates specified HRV metrics for each epoch in an MNE Epochs object.

        Args:
            epochs (mne.Epochs): MNE Epochs object. Events in epochs define the windows.
            rpeaks_samples (np.ndarray): Array of R-peak sample indices relative to the start of the continuous recording.
            original_sfreq (float): Sampling frequency of the original signal from which R-peaks were derived.
            metrics_to_calculate (Optional[List[str]]): List of HRV metrics to calculate (e.g., ['RMSSD', 'LFHF']).
                                                        Defaults to a predefined list if None.
            participant_id (Optional[str]): Participant ID to include in the output DataFrame.

        Returns:
            Optional[pd.DataFrame]: DataFrame with HRV metrics per epoch, or None on error.
                                    Columns: ['participant_id', 'condition', 'epoch_index', 'metric_name', 'value']
        """
        if epochs is None:
            self.logger.error("HRVAnalyzer (epoched): Epochs object is None. Cannot calculate HRV.")
            return None
        if rpeaks_samples is None or len(rpeaks_samples) < 2:
            self.logger.warning("HRVAnalyzer (epoched): Not enough R-peaks provided. Cannot calculate HRV.")
            return pd.DataFrame()
        if original_sfreq <= 0:
            self.logger.error("HRVAnalyzer (epoched): Invalid original_sfreq. Cannot calculate HRV.")
            return None

        metrics_list = metrics_to_calculate if metrics_to_calculate is not None else self.DEFAULT_EPOCHED_HRV_METRICS
        all_hrv_results = []

        for i, _ in enumerate(epochs): # Iterate using index to access epochs.events
            condition_name = "UnknownCondition"
            current_event_code = epochs.events[i, 2]
            for name, code in epochs.event_id.items():
                if code == current_event_code:
                    condition_name = name
                    break

            epoch_start_sample_orig = epochs.events[i, 0] + int(epochs.tmin * original_sfreq)
            epoch_end_sample_orig = epochs.events[i, 0] + int(epochs.tmax * original_sfreq)

            epoch_rpeaks = rpeaks_samples[(rpeaks_samples >= epoch_start_sample_orig) & (rpeaks_samples <= epoch_end_sample_orig)]

            if len(epoch_rpeaks) < 2:
                self.logger.debug(f"HRVAnalyzer (epoched): Less than 2 R-peaks in epoch {i} for condition '{condition_name}'. Skipping.")
                continue

            nn_intervals_epoch_ms = np.diff(epoch_rpeaks) / original_sfreq * 1000
            if len(nn_intervals_epoch_ms) < 1: continue

            try:
                hrv_indices = nk.hrv(nn_intervals_epoch_ms, sampling_rate=1000, show=False)
                for metric in metrics_list:
                    if metric in hrv_indices.columns:
                        all_hrv_results.append({
                            'participant_id': participant_id, 'condition': condition_name,
                            'epoch_index': i, 'metric_name': metric, 'value': hrv_indices[metric].iloc[0]
                        })
            except Exception as e:
                self.logger.error(f"HRVAnalyzer (epoched): Error calculating HRV for epoch {i}, condition '{condition_name}': {e}", exc_info=True)

        return pd.DataFrame(all_hrv_results) if all_hrv_results else pd.DataFrame()