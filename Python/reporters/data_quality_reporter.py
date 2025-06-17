import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import os
import json

class DataQualityReporter:
    """
    Generates reports on data quality for various modalities.
    """

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("DataQualityReporter initialized.")

    def report_eeg_quality(self,
                           raw_original: Optional[mne.io.Raw] = None,
                           raw_processed: Optional[mne.io.Raw] = None,
                           epochs_original_count: Optional[int] = None, # Number of events that *could* have become epochs
                           epochs_final: Optional[mne.Epochs] = None,
                           ica: Optional[mne.preprocessing.ICA] = None,
                           ica_labels: Optional[List[str]] = None, # List of labels for each ICA component
                           participant_id: Optional[str] = None
                           ) -> Dict[str, Any]:
        """
        Generates a dictionary of EEG data quality metrics.

        Args:
            raw_original (Optional[mne.io.Raw]): Original raw EEG data (before channel rejection).
            raw_processed (Optional[mne.io.Raw]): Processed raw EEG data (after bad channel interpolation, ICA).
            epochs_original_count (Optional[int]): Number of events that were candidates for epoching.
            epochs_final (Optional[mne.Epochs]): Final MNE Epochs object after all rejection.
            ica (Optional[mne.preprocessing.ICA]): Fitted ICA object.
            ica_labels (Optional[List[str]]): Labels for each ICA component (must match ica.n_components_).
            participant_id (Optional[str]): Participant ID for context.

        Returns:
            Dict[str, Any]: Dictionary of quality metrics.
        """
        metrics: Dict[str, Any] = {'participant_id': participant_id}
        self.logger.info(f"DataQualityReporter: Generating EEG quality report for P:{participant_id or 'N/A'}.")

        if raw_original:
            metrics['n_channels_original'] = len(raw_original.ch_names)
            metrics['original_sfreq'] = raw_original.info['sfreq']
            if raw_original.info.get('bads'):
                metrics['n_bad_channels_manual'] = len(raw_original.info['bads'])
                metrics['bad_channels_manual_names'] = raw_original.info['bads']

        if raw_processed:
            metrics['n_channels_processed'] = len(raw_processed.ch_names)
            # Bad channels after interpolation (should be 0 if interpolation worked)
            metrics['n_bad_channels_after_interp'] = len(raw_processed.info.get('bads', []))

        if ica:
            metrics['ica_n_components_fit'] = ica.n_components_
            if hasattr(ica, 'exclude') and ica.exclude:
                metrics['ica_n_components_rejected'] = len(ica.exclude)
                metrics['ica_rejected_indices'] = ica.exclude
                if ica_labels and len(ica_labels) == ica.n_components_:
                    rejected_comp_labels = [ica_labels[i] for i in ica.exclude]
                    metrics['ica_rejected_labels'] = rejected_comp_labels
                    # Count occurrences of each rejected label
                    label_counts = pd.Series(rejected_comp_labels).value_counts().to_dict()
                    metrics['ica_rejected_labels_counts'] = label_counts
            else:
                metrics['ica_n_components_rejected'] = 0

        if epochs_original_count is not None:
            metrics['n_epochs_original_candidate'] = epochs_original_count
            if epochs_final is not None:
                metrics['n_epochs_retained'] = len(epochs_final)
                if epochs_original_count > 0:
                    metrics['epoch_retention_rate'] = len(epochs_final) / epochs_original_count
                else:
                    metrics['epoch_retention_rate'] = np.nan if len(epochs_final) > 0 else 1.0 # Avoid 0/0

        return metrics

    def report_fnirs_quality(self,
                             raw_od: Optional[mne.io.Raw] = None,
                             raw_haemo: Optional[mne.io.Raw] = None,
                             # sci_values: Optional[Dict[str, float]] = None, # Example: {'ch_name': sci_value}
                             participant_id: Optional[str] = None
                             ) -> Dict[str, Any]:
        """Generates a dictionary of fNIRS data quality metrics."""
        metrics: Dict[str, Any] = {'participant_id': participant_id}
        self.logger.info(f"DataQualityReporter: Generating fNIRS quality report for P:{participant_id or 'N/A'}.")
        if raw_od:
            metrics['n_fnirs_od_channels'] = len(raw_od.ch_names)
        if raw_haemo:
            metrics['n_fnirs_haemo_channels'] = len(raw_haemo.ch_names)
            # Could add checks for NaN/Inf in raw_haemo.get_data()
        # if sci_values:
        #     metrics['avg_sci'] = np.mean(list(sci_values.values())) if sci_values else np.nan
        #     metrics['min_sci'] = np.min(list(sci_values.values())) if sci_values else np.nan
        return metrics

    def save_report_to_json(self, report_dict: Dict[str, Any], output_path: str) -> None:
        """Saves the quality report dictionary to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=4, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o) # Handle numpy types
            self.logger.info(f"DataQualityReporter: Report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"DataQualityReporter: Error saving report to {output_path}: {e}", exc_info=True)