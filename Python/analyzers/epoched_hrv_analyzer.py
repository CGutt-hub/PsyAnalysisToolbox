import mne
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import List, Optional, Dict, Any

class EpochedHRVAnalyzer:
    """
    Calculates HRV metrics for defined epochs or experimental conditions.
    """
    DEFAULT_HRV_METRICS = ['RMSSD', 'SDNN', 'pNN50', 'LF', 'HF', 'LFHF']

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EpochedHRVAnalyzer initialized.")

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
            self.logger.error("EpochedHRVAnalyzer: Epochs object is None. Cannot calculate HRV.")
            return None
        if rpeaks_samples is None or len(rpeaks_samples) < 2:
            self.logger.warning("EpochedHRVAnalyzer: Not enough R-peaks provided. Cannot calculate HRV.")
            return pd.DataFrame()
        if original_sfreq <= 0:
            self.logger.error("EpochedHRVAnalyzer: Invalid original_sfreq. Cannot calculate HRV.")
            return None

        metrics_list = metrics_to_calculate if metrics_to_calculate is not None else self.DEFAULT_HRV_METRICS
        all_hrv_results = []

        for i, epoch_data in enumerate(epochs): # MNE iterates over epochs, giving data arrays
            condition_name = "UnknownCondition"
            # Get condition name from epoch's event_id
            current_event_code = epochs.events[i, 2]
            for name, code in epochs.event_id.items():
                if code == current_event_code:
                    condition_name = name
                    break

            # Determine the time window of this epoch in samples of the original recording
            epoch_start_sample_orig = epochs.events[i, 0] + int(epochs.tmin * original_sfreq)
            epoch_end_sample_orig = epochs.events[i, 0] + int(epochs.tmax * original_sfreq)

            # Select R-peaks within this epoch's window
            epoch_rpeaks = rpeaks_samples[
                (rpeaks_samples >= epoch_start_sample_orig) & (rpeaks_samples <= epoch_end_sample_orig)
            ]

            if len(epoch_rpeaks) < 2: # Need at least 2 R-peaks for one NNI
                self.logger.debug(f"EpochedHRVAnalyzer: Less than 2 R-peaks in epoch {i} for condition '{condition_name}'. Skipping HRV for this epoch.")
                continue

            # Calculate NN-intervals in milliseconds for this epoch
            nn_intervals_epoch_ms = np.diff(epoch_rpeaks) / original_sfreq * 1000

            if len(nn_intervals_epoch_ms) < 1: # Need at least one NNI
                continue

            try:
                hrv_indices = nk.hrv(nn_intervals_epoch_ms, sampling_rate=1000, show=False) # Sampling rate for NNIs is effectively 1000Hz
                for metric in metrics_list:
                    if metric in hrv_indices.columns:
                        all_hrv_results.append({
                            'participant_id': participant_id,
                            'condition': condition_name,
                            'epoch_index': i,
                            'metric_name': metric,
                            'value': hrv_indices[metric].iloc[0]
                        })
            except Exception as e:
                self.logger.error(f"EpochedHRVAnalyzer: Error calculating HRV for epoch {i}, condition '{condition_name}': {e}", exc_info=True)

        return pd.DataFrame(all_hrv_results) if all_hrv_results else pd.DataFrame()