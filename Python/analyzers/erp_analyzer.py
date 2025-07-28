import mne
import numpy as np
import pandas as pd
from typing import Union # Added for Union type hint
from typing import Dict, List, Optional, Any, Tuple

class ERPAnalyzer:
    """
    Analyzer for Event-Related Potentials (ERP).
    Input: DataFrame (epochs/averages)
    Output: DataFrame
    """
    DEFAULT_FEATURE_MEASURES = ['peak_amplitude', 'mean_amplitude', 'peak_latency']

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ERPAnalyzer initialized.")

    def _reconstruct_epochs_from_df(self, data_df: pd.DataFrame, sfreq: float, ch_types_map: Dict[str, str]) -> Optional[mne.EpochsArray]:
        """
        Internal helper to reconstruct an MNE EpochsArray from a long-format DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame with columns ['epoch_id', 'condition', 'channel', 'time', 'value'].
            sfreq (float): The sampling frequency.
            ch_types_map (Dict[str, str]): A map of channel names to their types (e.g., {'Fz': 'eeg'}).

        Returns:
            Optional[mne.EpochsArray]: The reconstructed MNE Epochs object, or None on failure.
        """
        required_cols = ['epoch_id', 'condition', 'channel', 'time', 'value']
        if not all(col in data_df.columns for col in required_cols):
            self.logger.error(f"ERPAnalyzer: Input DataFrame is missing one or more required columns: {required_cols}")
            return None

        try:
            # --- 1. Prepare metadata for MNE objects ---
            ch_names = sorted(data_df['channel'].unique().tolist())
            if not all(ch in ch_types_map for ch in ch_names):
                missing = [ch for ch in ch_names if ch not in ch_types_map]
                self.logger.error(f"ERPAnalyzer: The following channels from the DataFrame are missing from 'ch_types_map': {missing}")
                return None

            times = np.sort(data_df['time'].unique())
            tmin = times[0]

            epoch_info = data_df[['epoch_id', 'condition']].drop_duplicates().sort_values('epoch_id').reset_index(drop=True)
            n_epochs = len(epoch_info)

            conditions_cat = epoch_info['condition'].astype('category')
            event_id_map = {name: i + 1 for i, name in enumerate(conditions_cat.cat.categories)}

            event_codes = epoch_info['condition'].map(event_id_map).to_numpy()
            events = np.array([np.arange(n_epochs), np.zeros(n_epochs, int), event_codes]).T

            # --- 2. Create the 3D data array (n_epochs, n_channels, n_times) ---
            data_pivoted = data_df.set_index(['epoch_id', 'channel', 'time'])['value'].unstack(level='time')

            epoch_ids_sorted = epoch_info['epoch_id'].to_numpy()
            full_multi_index = pd.MultiIndex.from_product([epoch_ids_sorted.tolist(), ch_names], names=['epoch_id', 'channel'])

            data_aligned = data_pivoted.reindex(full_multi_index)
            data_3d = data_aligned.to_numpy().reshape(n_epochs, len(ch_names), len(times))

            if np.isnan(data_3d).any(): # Changed is.nan to np.isnan for NumPy array check
                self.logger.warning("ERPAnalyzer: NaN values found after reconstructing 3D data array. This may indicate missing time points or channels for some epochs.")

            # --- 3. Create the final MNE objects ---
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=[ch_types_map[ch] for ch in ch_names]) # type: ignore
            epochs_array = mne.EpochsArray(data_3d, info, events=events, tmin=tmin, event_id=event_id_map, verbose=False)

            return epochs_array

        except Exception as e:
            self.logger.error(f"ERPAnalyzer: Failed to reconstruct MNE Epochs from DataFrame: {e}", exc_info=True)
            return None

    def calculate_component_features_from_df(self,
                                             epochs_df: pd.DataFrame,
                                             sfreq: float,
                                             ch_types_map: Dict[str, str],
                                             component_configs: Dict[str, Dict[str, Any]],
                                             participant_id: Optional[str] = None
                                             ) -> Optional[pd.DataFrame]:
        """
        Calculates features for specified ERP components from a long-format DataFrame.

        Args:
            epochs_df (pd.DataFrame): DataFrame in long format with required columns:
                                      ['epoch_id', 'condition', 'channel', 'time', 'value'].
            sfreq (float): The sampling frequency of the data.
            ch_types_map (Dict[str, str]): A map of channel names to their types (e.g., {'Fz': 'eeg'}).
            component_configs (Dict[str, Dict[str, Any]]): Configuration for ERP components.
            participant_id (Optional[str]): Participant ID to include in the output DataFrame.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ERP features, or None on error.
        """
        self.logger.info("ERPAnalyzer: Attempting to calculate ERP features from DataFrame.")
        epochs_from_df = self._reconstruct_epochs_from_df(epochs_df, sfreq, ch_types_map)
        if epochs_from_df is None:
            self.logger.error("ERPAnalyzer: Could not reconstruct Epochs object from DataFrame. Aborting feature calculation.")
            return None

        return self.calculate_component_features(
            epochs=epochs_from_df,
            component_configs=component_configs,
            participant_id=participant_id
        )

    def calculate_component_features(self,
                                     epochs: Union[mne.Epochs, mne.EpochsArray],
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
                    # Ensure channels is a list for pick method
                    channels_list = channels if isinstance(channels, list) else [channels]
                    
                    evoked_roi = evoked.copy().pick(channels_list).crop(tmin=tmin, tmax=tmax) # type: ignore
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

    def get_averaged_erps_from_df(self,
                                  epochs_df: pd.DataFrame,
                                  sfreq: float,
                                  ch_types_map: Dict[str, str]
                                  ) -> Dict[str, mne.Evoked]:
        """
        Reconstructs MNE Epochs from a DataFrame and returns averaged Evoked objects.

        Args:
            epochs_df (pd.DataFrame): DataFrame with columns ['epoch_id', 'condition', 'channel', 'time', 'value'].
            sfreq (float): The sampling frequency of the data.
            See `calculate_component_features_from_df` for an explanation of the arguments.

        Returns:
            Dict[str, mne.Evoked]: A dictionary of MNE Evoked objects, one for each condition.
        """
        epochs_from_df = self._reconstruct_epochs_from_df(epochs_df, sfreq, ch_types_map)
        return self.get_averaged_erps(epochs=epochs_from_df) if epochs_from_df else {}

    def get_averaged_erps(self, epochs: Union[mne.Epochs, mne.EpochsArray]) -> Dict[str, mne.Evoked]:
        """Returns a dictionary of MNE Evoked objects, one for each condition."""
        if epochs is None:
            self.logger.error("ERPAnalyzer: Epochs object is None. Cannot get averaged ERPs.")
            return {}
        return {condition: epochs[condition].average() for condition in epochs.event_id.keys() if len(epochs[condition]) > 0} # type: ignore