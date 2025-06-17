import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

class ERPAnalyzer:
    """
    Analyzes Event-Related Potentials (ERPs) from epoched EEG data.
    """
    DEFAULT_FEATURE_MEASURES = ['peak_amplitude', 'mean_amplitude', 'peak_latency']

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ERPAnalyzer initialized.")

    def calculate_component_features(self,
                                     epochs: mne.Epochs,
                                     component_configs: Dict[str, Dict[str, Any]],
                                     participant_id: Optional[str] = None
                                     ) -> Optional[pd.DataFrame]:
        """
        Calculates features for specified ERP components.

        Args:
            epochs (mne.Epochs): Epoched EEG data.
            component_configs (Dict[str, Dict[str, Any]]): Configuration for ERP components.
                Example: {
                    'P300': {'tmin': 0.25, 'tmax': 0.50, 'channels': ['Pz', 'Cz'], 'measure': ['peak_amplitude', 'peak_latency']},
                    'N170': {'tmin': 0.13, 'tmax': 0.20, 'channels': ['P7', 'P8'], 'measure': ['mean_amplitude']}
                }
                'measure' can be a string or list from ['peak_amplitude', 'mean_amplitude', 'peak_latency'].
            participant_id (Optional[str]): Participant ID to include in the output DataFrame.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ERP features in long format, or None on error.
                                    Columns: ['participant_id', 'condition', 'component', 'channel_roi',
                                              'feature', 'value']
        """
        if epochs is None:
            self.logger.error("ERPAnalyzer: Epochs object is None. Cannot calculate component features.")
            return None
        if not component_configs:
            self.logger.warning("ERPAnalyzer: component_configs is empty. No features to calculate.")
            return pd.DataFrame()

        all_features_list = []

        for condition_name in epochs.event_id.keys():
            try:
                condition_epochs = epochs[condition_name]
                if len(condition_epochs) == 0:
                    self.logger.info(f"ERPAnalyzer: No epochs for condition '{condition_name}'. Skipping.")
                    continue

                evoked = condition_epochs.average() # Average epochs for this condition

                for comp_name, comp_config in component_configs.items():
                    tmin = comp_config.get('tmin')
                    tmax = comp_config.get('tmax')
                    channels = comp_config.get('channels')
                    measures = comp_config.get('measure', self.DEFAULT_FEATURE_MEASURES)
                    if isinstance(measures, str):
                        measures = [measures]

                    if tmin is None or tmax is None or not channels:
                        self.logger.warning(f"ERPAnalyzer: Invalid config for component '{comp_name}' (missing tmin, tmax, or channels). Skipping.")
                        continue

                    # Create a temporary evoked object for the ROI and time window
                    evoked_roi = evoked.copy().pick(channels).crop(tmin=tmin, tmax=tmax)
                    data_roi_avg = evoked_roi.data.mean(axis=0) # Average over selected channels for ROI
                    times_roi = evoked_roi.times

                    for measure in measures:
                        value = np.nan
                        if measure == 'peak_amplitude':
                            peak_idx = np.argmax(np.abs(data_roi_avg))
                            value = data_roi_avg[peak_idx]
                        elif measure == 'mean_amplitude':
                            value = np.mean(data_roi_avg)
                        elif measure == 'peak_latency':
                            peak_idx = np.argmax(np.abs(data_roi_avg))
                            value = times_roi[peak_idx]
                        else:
                            self.logger.warning(f"ERPAnalyzer: Unknown measure '{measure}' for component '{comp_name}'. Skipping.")
                            continue

                        all_features_list.append({
                            'participant_id': participant_id,
                            'condition': condition_name,
                            'component': comp_name,
                            'channel_roi': "_".join(channels) if isinstance(channels, list) else channels,
                            'feature': measure,
                            'value': value
                        })
            except Exception as e:
                self.logger.error(f"ERPAnalyzer: Error processing condition '{condition_name}': {e}", exc_info=True)

        return pd.DataFrame(all_features_list) if all_features_list else pd.DataFrame()

    def get_averaged_erps(self, epochs: mne.Epochs) -> Dict[str, mne.Evoked]:
        """Returns a dictionary of MNE Evoked objects, one for each condition."""
        if epochs is None:
            self.logger.error("ERPAnalyzer: Epochs object is None. Cannot get averaged ERPs.")
            return {}
        return {condition: epochs[condition].average() for condition in epochs.event_id.keys() if len(epochs[condition]) > 0}