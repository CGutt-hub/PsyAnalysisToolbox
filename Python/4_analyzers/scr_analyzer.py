import mne
import neurokit2 as nk
import numpy as np
import pandas as pd # Added for DataFrame type hint
from typing import Dict, Optional # For type hinting

class SCRAnalyzer:
    def __init__(self, logger):
        self.logger = logger # type: ignore # Logger is expected to be set by the caller
        self.logger.info("SCRAnalyzer initialized.")

    def calculate_eda_features_per_condition(self, raw_eeg_with_events, # Used for event timings
                                             phasic_eda_full_signal_array: np.ndarray, # Full phasic EDA signal (input for mean phasic and SCR detection)
                                             scr_features_df: Optional[pd.DataFrame], # DataFrame of SCR features from EDASCRProcessor
                                             eda_original_sfreq: float, # Original sampling rate of EDA
                                             stimulus_duration_seconds: float # Duration of the stimulus/epoch
                                             ) -> Dict[str, float]:
        """
        Calculates EDA features (e.g., mean phasic amplitude, SCR count) per condition.
        Specifically calculates mean phasic amplitude and SCR count within stimulus epochs.
        using pre-detected SCRs.
        
        Args:
            raw_eeg_with_events (mne.io.Raw): Raw EEG object containing event annotations.
            phasic_eda_full_signal_array (np.ndarray): The full preprocessed phasic EDA signal.
            scr_features_df (Optional[pd.DataFrame]): DataFrame containing features of detected SCRs.
                                                     Expected columns include 'SCR_Onsets_Time'.
            eda_original_sfreq (float): Original sampling rate of the EDA signal.
            stimulus_duration_seconds (float): The duration of the stimulus epoch in seconds.
            
        Returns:
            Dict[str, float]: A dictionary where keys are feature names (e.g., 'eda_phasic_mean_ConditionName')
                              and values are the calculated feature values. Returns an empty dict on failure.
        """
        calculated_metrics: Dict[str, float] = {}

        if raw_eeg_with_events is None or raw_eeg_with_events.info['sfreq'] is None:
            self.logger.warning("SCRAnalyzer - No EEG data with events or sampling frequency provided. Skipping EDA feature extraction.")
            return calculated_metrics
        if phasic_eda_full_signal_array is None or eda_original_sfreq is None:
            self.logger.warning("SCRAnalyzer - Phasic EDA signal array or original sampling rate not provided. Skipping.")
            return calculated_metrics
        if scr_features_df is None:
            self.logger.info("SCRAnalyzer - SCR features DataFrame not provided. SCR count will not be calculated.")
            return calculated_metrics
        if stimulus_duration_seconds is None or stimulus_duration_seconds <= 0:
            self.logger.warning("SCRAnalyzer - Invalid stimulus_duration_seconds provided. Skipping.")
            return calculated_metrics
        if eda_original_sfreq <= 0:
            self.logger.warning("SCRAnalyzer - Invalid eda_original_sfreq provided. Skipping.")
            return calculated_metrics

        self.logger.info("SCRAnalyzer - Calculating EDA features per condition.")
        try:
            events, event_id_map = mne.events_from_annotations(raw_eeg_with_events, verbose=False)

            if not events.size:
                self.logger.warning("SCRAnalyzer - No events found in EEG data. Cannot extract condition-specific EDA features.")
                return calculated_metrics

            for condition_name, event_code in event_id_map.items():
                self.logger.debug(f"SCRAnalyzer - Processing condition: {condition_name}")
                condition_event_indices = events[events[:, 2] == event_code, 0] # Get sample onsets
                condition_scr_counts = []
                condition_phasic_means = []

                for onset_sample in condition_event_indices:
                    start_time_sec = onset_sample / raw_eeg_with_events.info['sfreq']
                    end_time_sec = start_time_sec + stimulus_duration_seconds

                    # Extract corresponding segment from phasic_eda_full
                    start_idx_eda = int(start_time_sec * eda_original_sfreq)
                    end_idx_eda = int(end_time_sec * eda_original_sfreq)

                    if start_idx_eda < end_idx_eda and end_idx_eda <= len(phasic_eda_full_signal_array) and start_idx_eda >= 0: # Ensure indices are valid
                        eda_epoch = phasic_eda_full_signal_array[start_idx_eda:end_idx_eda]
                        if len(eda_epoch) > 0:
                            # Example features using NeuroKit2
                            # For mean phasic, directly average the phasic component
                            condition_phasic_means.append(np.mean(eda_epoch))
                            
                            # Calculate SCR count for the epoch using pre-detected SCRs
                            if scr_features_df is not None and not scr_features_df.empty and 'SCR_Onsets_Time' in scr_features_df.columns:
                                epoch_scrs = scr_features_df[
                                    (scr_features_df['SCR_Onsets_Time'] >= start_time_sec) &
                                    (scr_features_df['SCR_Onsets_Time'] < end_time_sec)
                                ]
                                condition_scr_counts.append(len(epoch_scrs))
                            elif scr_features_df is not None and ('SCR_Onsets_Time' not in scr_features_df.columns and not scr_features_df.empty):
                                self.logger.warning(f"SCRAnalyzer - 'SCR_Onsets_Time' column missing in scr_features_df. Cannot count SCRs for condition '{condition_name}'.")
                        else:
                             self.logger.debug(f"SCRAnalyzer - Extracted EDA epoch for condition '{condition_name}' at time {start_time_sec:.2f}s is empty.")
                    else:
                         self.logger.debug(f"SCRAnalyzer - Invalid EDA segment indices for condition '{condition_name}' at time {start_time_sec:.2f}s. Start: {start_idx_eda}, End: {end_idx_eda}, Signal Len: {len(phasic_eda_full_signal_array)}")
                # Calculate mean phasic amplitude and mean SCR count for the condition
                if condition_phasic_means: calculated_metrics[f'eda_phasic_mean_{condition_name}'] = np.nanmean(condition_phasic_means)
                if condition_scr_counts: calculated_metrics[f'eda_scr_count_{condition_name}'] = np.nanmean(condition_scr_counts)
            self.logger.info("SCRAnalyzer - Condition-specific EDA feature calculation completed.")
        except Exception as e: # Catch any unexpected errors during the process
            self.logger.error(f"SCRAnalyzer - Error calculating EDA features: {e}", exc_info=True)
        
        return calculated_metrics