import mne
import numpy as np
from typing import Dict, List, Tuple, Optional, Any # Added Any for welch_params
# No direct import of FAIAnalyzer here, it will be passed as an instance

class PSDAnalyzer:
    # Default parameters for Welch's method
    DEFAULT_WELCH_WINDOW = 'hann'
    DEFAULT_WELCH_N_FFT_SECONDS = 1.0  # For deriving n_fft from sfreq
    DEFAULT_WELCH_OVERLAP_RATIO = 0.5 # For deriving n_overlap from n_fft
    DEFAULT_SKIP_CONDITIONS_PSD = ["bad_stim", "boundary", "edge"] # Lowercase for case-insensitive check

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("PSDAnalyzer initialized.")

    def _calculate_psd_for_epochs(self, epochs: mne.Epochs, sfreq: float,
                                  band_freqs: Tuple[float, float],
                                  welch_params_custom: Optional[Dict[str, Any]] = None,
                                  condition_name_logging: str = ""):
        """Helper to compute PSD for given epochs and band."""

        # Default Welch parameters
        default_n_fft = int(sfreq * self.DEFAULT_WELCH_N_FFT_SECONDS)
        # Use epoch duration if it's shorter than 1 second for n_fft
        epoch_duration_samples = epochs.times.shape[0] if epochs.times is not None else 0
        if epoch_duration_samples > 0 and epoch_duration_samples < default_n_fft:
             default_n_fft = epoch_duration_samples # Use epoch length if shorter than 1s

        default_n_overlap = int(default_n_fft * self.DEFAULT_WELCH_OVERLAP_RATIO)

        current_welch_params = {
            'n_fft': default_n_fft,
            'n_overlap': default_n_overlap,
            'window': self.DEFAULT_WELCH_WINDOW
        }

        if welch_params_custom: # User provided overrides
            if 'n_fft' in welch_params_custom and welch_params_custom['n_fft'] is not None:
                current_welch_params['n_fft'] = int(welch_params_custom['n_fft'])
            # else it keeps the default_n_fft derived from sfreq

            # n_overlap: if user provides n_overlap, use it.
            # Else, base overlap on the current n_fft (either user-specified or default).
            if 'n_overlap' in welch_params_custom and welch_params_custom['n_overlap'] is not None:
                user_n_overlap = int(welch_params_custom['n_overlap'])
                 # Ensure user-provided n_overlap doesn't exceed n_fft
                if user_n_overlap > current_welch_params['n_fft']:
                     self.logger.warning(f"PSDAnalyzer - User-specified n_overlap ({user_n_overlap}) exceeds n_fft ({current_welch_params['n_fft']}). Using n_fft for n_overlap.")
                     current_welch_params['n_overlap'] = current_welch_params['n_fft']
                else:
                    current_welch_params['n_overlap'] = user_n_overlap
            else:
                current_welch_params['n_overlap'] = int(current_welch_params['n_fft'] * self.DEFAULT_WELCH_OVERLAP_RATIO)

            if 'window' in welch_params_custom and welch_params_custom['window'] is not None:
                current_welch_params['window'] = welch_params_custom['window']

        # Ensure n_fft is at least 1
        if current_welch_params['n_fft'] < 1:
             self.logger.warning(f"PSDAnalyzer - Calculated n_fft is less than 1 ({current_welch_params['n_fft']}). Setting to 1.")
             current_welch_params['n_fft'] = 1
        # Ensure n_overlap is not greater than n_fft
        if current_welch_params['n_overlap'] > current_welch_params['n_fft']:
             self.logger.warning(f"PSDAnalyzer - Calculated n_overlap ({current_welch_params['n_overlap']}) exceeds n_fft ({current_welch_params['n_fft']}). Setting n_overlap = n_fft.")
             current_welch_params['n_overlap'] = current_welch_params['n_fft']
        # Ensure n_overlap is not negative
        if current_welch_params['n_overlap'] < 0:
             self.logger.warning(f"PSDAnalyzer - Calculated n_overlap is negative ({current_welch_params['n_overlap']}). Setting to 0.")
             current_welch_params['n_overlap'] = 0

        try:
            # compute_psd expects fmin, fmax, and other welch params directly
            psd_obj = epochs.compute_psd(
                method='welch',
                fmin=band_freqs[0], # type: ignore[arg-type]
                fmax=band_freqs[1],
                n_fft=current_welch_params['n_fft'],
                n_overlap=current_welch_params['n_overlap'],
                window=current_welch_params['window'],
                average='mean', # Enforce 'mean' for this helper's purpose of getting (n_channels, n_freqs)
                verbose=False
            )
            # psds_data shape is (n_channels, n_freqs) when average='mean'
            psds_data, _ = psd_obj.get_data(return_freqs=True) # get_data() returns a tuple (data, freqs)

            # We want mean power per channel across the frequency band
            mean_power_per_channel = np.mean(psds_data, axis=1)
            return mean_power_per_channel
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Error computing PSD for band {band_freqs} {condition_name_logging}: {e}", exc_info=True)
            return None

    def compute_psd_per_condition(self,
                                  epochs_all_conditions: mne.Epochs, # Changed input
                                  bands_config: Dict[str, Tuple[float, float]],
                                #   analysis_epoch_tmax_config: float, # No longer needed here
                                #   channels_to_pick: List[str], # Channels are in the epochs object
                                  welch_params_config: Optional[Dict[str, Any]] = None
                                  ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Computes Power Spectral Density (PSD) for specified bands and channels, per condition,
        using a pre-constructed MNE Epochs object that contains all relevant conditions.
        """
        psd_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        if epochs_all_conditions is None or len(epochs_all_conditions) == 0: # Check if epochs object is valid
            self.logger.warning("PSDAnalyzer - Epochs object is None or empty. Cannot compute PSD per condition.")
            return psd_results

        sfreq = epochs_all_conditions.info['sfreq']

        for condition_name in epochs_all_conditions.event_id.keys(): # Iterate through conditions in the Epochs object
            if condition_name.lower() in self.DEFAULT_SKIP_CONDITIONS_PSD:
                self.logger.debug(f"PSDAnalyzer - Skipping PSD for non-experimental condition '{condition_name}'.")
                continue

            self.logger.debug(f"PSDAnalyzer - Processing PSD for condition: {condition_name}")
            psd_results[condition_name] = {band_name: {} for band_name in bands_config.keys()}

            try:
                epochs = epochs_all_conditions[condition_name] # Select epochs for the current condition

                if len(epochs) == 0:
                    self.logger.info(f"PSDAnalyzer - No epochs for condition '{condition_name}' for PSD calculation.")
                    continue

                for band_name, band_freq_range in bands_config.items():
                    power_per_channel_for_band = self._calculate_psd_for_epochs(
                        epochs, sfreq, band_freq_range, welch_params_config,
                        f"Cond: {condition_name}" # Band info is implicit in band_freq_range
                    )
                    if power_per_channel_for_band is not None:
                        for ch_idx, ch_name_in_epoch in enumerate(epochs.ch_names):
                            psd_results[condition_name][band_name][ch_name_in_epoch] = power_per_channel_for_band[ch_idx]

            except Exception as e_cond_psd:
                self.logger.error(f"PSDAnalyzer - Error computing PSD for condition '{condition_name}': {e_cond_psd}", exc_info=True)

        return psd_results

    def calculate_psd_from_epochs(self,
                              epochs_processed_all_conditions: Optional[mne.Epochs], # Changed input
                              bands_config: Dict[str, Tuple[float, float]],
                              psd_channels_of_interest: Optional[List[str]] = None,
                              welch_params_config: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculates Power Spectral Density (PSD) for specified bands and channels from MNE Epochs.

        Args:
            epochs_processed_all_conditions (mne.Epochs): MNE Epochs object containing data for all relevant conditions,
                                                          potentially with metadata from EPrimePreprocessor.
            bands_config (dict): Dictionary mapping band names to (fmin, fmax) tuples (e.g., {'Alpha': (8,12)}).
            psd_channels_of_interest (list, optional): Specific list of channel names for PSD calculation.
                                                       If None, PSD is calculated for all channels in the epochs.
            welch_params_config (dict, optional): Parameters for Welch's method (e.g., {'n_fft': 1024, 'n_overlap': 512, 'window': 'hamming'}).
                                                  If None, defaults are used (1s window, 50% overlap, hann window).

        Returns:
            dict: psd_results in the format {'condition_name': {'band_name': {'channel_name': power_value}}}
        """
        psd_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        if epochs_processed_all_conditions is None or len(epochs_processed_all_conditions) == 0:
            self.logger.warning("PSDAnalyzer - No processed EEG epochs provided or epochs object is empty. Skipping PSD calculation.")
            return psd_results
        if not bands_config or not isinstance(bands_config, dict):
            self.logger.warning("PSDAnalyzer - 'bands_config' must be a non-empty dictionary. Skipping.")
            return psd_results

        # Determine channels for PSD calculation
        final_psd_picks = []
        if psd_channels_of_interest:
            final_psd_picks = [ch for ch in psd_channels_of_interest if ch in epochs_processed_all_conditions.ch_names]
            if not final_psd_picks:
                self.logger.warning(f"PSDAnalyzer - None of the specified 'psd_channels_of_interest' found in EEG data: {psd_channels_of_interest}. Skipping PSD.")
                return psd_results
        else: # No specific channels, no FAI -> calculate for all channels
            self.logger.info("PSDAnalyzer - No specific PSD channels or FAI pairs defined. Calculating PSD for all available channels.")
            final_psd_picks = epochs_processed_all_conditions.ch_names

        if not final_psd_picks:
             self.logger.warning(f"PSDAnalyzer - No channels selected for PSD calculation. Skipping.")
             return psd_results

        self.logger.info(f"PSDAnalyzer - Calculating PSD for channels: {final_psd_picks}")

        try:
            # If specific channels were requested for PSD, create a new epochs object with only those channels
            # Otherwise, use all channels present in the input epochs object
            # Use .copy() to avoid modifying the original epochs object passed in
            epochs_for_psd = epochs_processed_all_conditions.copy().pick(final_psd_picks, verbose=False)
            if len(epochs_for_psd.ch_names) == 0:
                self.logger.warning(f"PSDAnalyzer - No channels remained after picking for PSD: {final_psd_picks}. Skipping.")
                return psd_results

            # Step 1: Compute PSD for all relevant conditions and bands
            psd_results = self.compute_psd_per_condition(
                epochs_all_conditions=epochs_for_psd, # Use the (potentially channel-subsetted) epochs
                bands_config=bands_config,
                welch_params_config=welch_params_config
            )

            self.logger.info("PSDAnalyzer - PSD calculation completed.")
            return psd_results
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Critical error calculating PSD: {e}", exc_info=True)
            return psd_results # Return whatever was computed so far
