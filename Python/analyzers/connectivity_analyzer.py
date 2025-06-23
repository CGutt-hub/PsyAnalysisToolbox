import mne
from mne_connectivity import SpectralConnectivity, spectral_connectivity_epochs
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import List, Optional, Union, Tuple, Dict, Any, cast # Added cast

# Utility function for PLV calculation on segments
def _calculate_plv_segment(phase_sig1: np.ndarray, phase_sig2: np.ndarray, logger_obj) -> float:
    min_len = min(len(phase_sig1), len(phase_sig2))
    if min_len == 0:
        logger_obj.debug("ConnectivityAnalyzer (PLV segment): Zero length signal provided.")
        return np.nan
    phase_diff = phase_sig1[:min_len] - phase_sig2[:min_len]
    return np.abs(np.mean(np.exp(1j * phase_diff)))

class ConnectivityAnalyzer:
    """
    Computes EEG-EEG functional connectivity measures.
    """
    DEFAULT_CONN_MODE = 'multitaper'
    DEFAULT_FAVERAGE = True
    
    # Default parameters for epoched-vs-continuous PLV
    DEFAULT_TRIAL_IDENTIFIER_EPRIME = "N/A_TRIAL_ID"
    OUTPUT_COL_PARTICIPANT_ID = 'participant_id'
    OUTPUT_COL_CONDITION = 'condition'
    OUTPUT_COL_EPOCH_INDEX = 'epoch_index_overall'
    OUTPUT_COL_TRIAL_ID_EPRIME = 'trial_identifier_eprime'
    OUTPUT_COL_MODALITY_PAIR = 'modality_pair'
    OUTPUT_COL_SIGNAL1_BAND = 'signal1_band' # More generic
    OUTPUT_COL_PLV_VALUE = 'plv_value' # More generic

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ConnectivityAnalyzer initialized.")

    def calculate_spectral_connectivity_epochs(self,
                                               epochs: mne.Epochs,
                                               method: Union[str, List[str]],
                                               mode: str = DEFAULT_CONN_MODE,
                                               fmin: Optional[Union[float, Tuple[float, ...]]] = None,
                                               fmax: Optional[Union[float, Tuple[float, ...]]] = None,
                                               faverage: bool = DEFAULT_FAVERAGE,
                                               tmin: Optional[float] = None,
                                               tmax: Optional[float] = None,
                                               mt_bandwidth: Optional[float] = None,
                                               mt_low_bias: bool = True,
                                               picks: Optional[List[str]] = None,
                                               indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                               n_jobs: int = 1
                ) -> Optional[Union[SpectralConnectivity, List[SpectralConnectivity]]]:
        """
        Wrapper for mne.connectivity.spectral_connectivity using epochs.

        Args:
            epochs (mne.Epochs): Epoched EEG data.
            method (Union[str, List[str]]): Connectivity measure(s) to compute (e.g., 'coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased').
            mode (str): Spectrum estimation mode, e.g., 'multitaper', 'fourier', 'cwt_morlet'.
            fmin (Optional[Union[float, Tuple[float, ...]]]): Lower frequency or frequencies of interest.
            fmax (Optional[Union[float, Tuple[float, ...]]]): Upper frequency or frequencies of interest.
            faverage (bool): Average over frequencies.
            tmin (Optional[float]): Time window start for connectivity estimation.
            tmax (Optional[float]): Time window end.
            mt_bandwidth (Optional[float]): Multitaper bandwidth.
            mt_low_bias (bool): Multitaper low bias.
            picks (Optional[List[str]]): Channels to include. If None, uses all EEG channels.
            indices (Optional[Tuple[np.ndarray, np.ndarray]]): Optional precomputed channel indices for connectivity.
            n_jobs (int): Number of jobs for parallel computation.

        Returns:
            Optional[Union[mne.connectivity.SpectralConnectivity, List[mne.connectivity.SpectralConnectivity]]]:
            The MNE SpectralConnectivity object(s), or None on error.
        """
        if epochs is None:
            self.logger.error("ConnectivityAnalyzer: Epochs object is None. Cannot calculate connectivity.")
            return None

        # Handle channel picking: convert user-provided names to indices if necessary
        # If 'picks' is None, mne.pick_types will select all EEG channels by default.
        # If 'picks' is a list of strings, convert them to indices.
        # If 'picks' is already an array of indices, use it directly.
        actual_picks_for_conn = None
        if picks is not None:
            try:
                actual_picks_for_conn = mne.pick_channels(epochs.ch_names, include=picks, ordered=True)
            except ValueError as e:
                self.logger.error(f"ConnectivityAnalyzer: Error picking channels from user-provided list '{picks}': {e}. Cannot calculate connectivity.")
                return None
        else:
            actual_picks_for_conn = mne.pick_types(epochs.info, eeg=True) # type: ignore[reportArgumentType] # Pick all EEG channels if no specific picks

        if actual_picks_for_conn is None or actual_picks_for_conn.size == 0:
            self.logger.error("ConnectivityAnalyzer: No EEG channels found or selected. Cannot calculate connectivity.")
            return None


        self.logger.info(f"ConnectivityAnalyzer: Calculating spectral connectivity with method(s) '{method}', mode '{mode}'.")
        try:
            # spectral_connectivity_epochs returns Union[SpectralConnectivity, List[SpectralConnectivity]]
            # Pylance seems to be misinterpreting this.
            result = spectral_connectivity_epochs(
                epochs, method=method, mode=mode, indices=indices, # type: ignore
                sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax, faverage=faverage, # type: ignore
                tmin=tmin, tmax=tmax, mt_bandwidth=mt_bandwidth, mt_low_bias=mt_low_bias,
                n_jobs=n_jobs, verbose=False, picks=actual_picks_for_conn # type: ignore[reportCallIssue]
            )
            # Explicitly cast the result to guide Pylance.
            # This cast assumes `result` is not None, which is true if no exception was raised.
            if isinstance(method, list):
                return cast(List[SpectralConnectivity], result)
            else:
                return cast(SpectralConnectivity, result)
        except Exception as e:
            self.logger.error(f"ConnectivityAnalyzer: Error calculating spectral connectivity: {e}", exc_info=True)
            return None
        
    def calculate_epoched_vs_continuous_plv(self,
                                            signal1_epochs: mne.Epochs,
                                            signal1_channels_to_average: List[str],
                                            signal1_bands_config: Dict[str, Tuple[float, float]],
                                            signal1_original_sfreq_for_event_timing: float,
                                            signal2_continuous_data: np.ndarray,
                                            signal2_sfreq: float,
                                            signal1_name: str, # e.g., "EEG"
                                            signal2_name: str, # e.g., "HRV" or "EDA"
                                            participant_id: str
                                            ) -> pd.DataFrame:
        """
        Calculates trial-wise PLV between an epoched signal (averaged over channels and filtered into bands)
        and a continuous signal.
        This is achieved by creating a parallel Epochs object for the continuous
        signal, ensuring robust time alignment and resampling.

        Args:
            signal1_epochs (mne.Epochs): Epoched data for the first signal.
            signal1_channels_to_average (list): List of channel names from signal1_epochs to average.
            signal1_bands_config (dict): Dictionary mapping band names to (fmin, fmax) for signal1.
            signal1_original_sfreq_for_event_timing (float): Original sampling rate of the raw data
                                                             from which signal1_epochs were derived.
            signal2_continuous_data (np.ndarray): Full continuous data for the second signal.
            signal2_sfreq (float): Sampling frequency of signal2_continuous_data.
            signal1_name (str): Name for the first signal (e.g., "EEG").
            signal2_name (str): Name for the second signal (e.g., "HRV", "EDA").
            participant_id (str): Participant ID.

        Returns:
            pd.DataFrame: DataFrame with trial-wise PLV results.
        """
        if signal1_epochs is None or not signal1_channels_to_average:
            self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Signal 1 epochs or channels to average not provided for P:{participant_id}. Skipping PLV for {signal1_name}-{signal2_name}.")
            return pd.DataFrame()
        if signal2_continuous_data is None or signal2_sfreq is None:
            self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Signal 2 continuous data or sfreq not provided for P:{participant_id}. Skipping PLV for {signal1_name}-{signal2_name}.")
            return pd.DataFrame()

        self.logger.info(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Starting trial-wise PLV for P:{participant_id} between {signal1_name} (channels: {signal1_channels_to_average}) and {signal2_name}.")

        # --- Create a parallel MNE Epochs object for the continuous signal (signal2) ---
        # 1. Wrap continuous data in an MNE Raw object
        ch_name_s2 = [f"{signal2_name}_continuous"]
        info_s2 = mne.create_info(ch_names=ch_name_s2, sfreq=signal2_sfreq, ch_types='misc')
        raw_s2 = mne.io.RawArray(signal2_continuous_data.reshape(1, -1), info_s2, verbose=False)

        # 2. Adjust event sample timings to match signal2's sampling rate
        events_s1 = signal1_epochs.events
        events_s2 = events_s1.copy()
        events_s2[:, 0] = np.round(events_s1[:, 0] * (signal2_sfreq / signal1_original_sfreq_for_event_timing)).astype(int)

        # 3. Create epochs from the continuous signal using the adjusted events
        try:
            signal2_epochs = mne.Epochs(raw_s2, events_s2, event_id=signal1_epochs.event_id,
                                        tmin=signal1_epochs.tmin, tmax=signal1_epochs.tmax,
                                        baseline=None, preload=True, verbose=False)
        except ValueError as e:
            self.logger.error(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Error creating epochs for signal 2 for P:{participant_id}. This can happen if event timings fall outside the continuous signal duration. Error: {e}")
            return pd.DataFrame()

        # 4. Resample signal2_epochs to match signal1_epochs' sampling rate for perfect alignment
        all_trial_plv_results = []
        signal1_epoch_sfreq = signal1_epochs.info['sfreq']
        signal2_epochs.resample(signal1_epoch_sfreq, verbose=False)
        # --- End of Epochs creation ---

        if not signal1_bands_config:
            self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Signal 1 bands configuration is empty for P:{participant_id}. Skipping PLV calculation for {signal1_name}-{signal2_name}.")
            return pd.DataFrame()

        # Iterate through both sets of epochs simultaneously
        for i, (epoch_s1_data, epoch_s2_data) in enumerate(zip(signal1_epochs, signal2_epochs)):
            current_epoch_data_s1: np.ndarray = cast(np.ndarray, epoch_s1_data)

            # Get condition name from event_id
            current_event_details = signal1_epochs.events[i] # Array: [sample, previous_event_id, event_id]
            condition_code = current_event_details[2]
            condition_name = next((name for name, code in signal1_epochs.event_id.items() if code == condition_code), "UnknownCondition")

            # Pick and average channels for signal 1
            try:
                missing_channels = [ch for ch in signal1_channels_to_average if ch not in signal1_epochs.ch_names]
                if missing_channels:
                    self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Channels {missing_channels} not found in signal1_epochs for P:{participant_id}, trial {i}, cond {condition_name}. Skipping trial.")
                    continue
                
                channel_indices_for_avg = mne.pick_channels(signal1_epochs.ch_names, include=signal1_channels_to_average, ordered=True)
                if len(channel_indices_for_avg) == 0:
                    self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): No channels selected from signal1_channels_to_average for P:{participant_id}, trial {i}, cond {condition_name}. Skipping.")
                    continue
                signal1_trial_data_avg = current_epoch_data_s1[channel_indices_for_avg, :].mean(axis=0).ravel()
            except Exception as e_pick:
                self.logger.error(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Error picking channels for signal 1 in trial {i}, P:{participant_id}, cond {condition_name}: {e_pick}. Skipping.")
                continue

            # Get aligned data for signal 2 from its epochs object
            signal2_segment_for_plv = cast(np.ndarray, epoch_s2_data).ravel()

            # Get trial identifier from metadata if available
            trial_identifier_eprime_str = self.DEFAULT_TRIAL_IDENTIFIER_EPRIME
            if signal1_epochs.metadata is not None and 'trial_identifier_eprime' in signal1_epochs.metadata.columns:
                trial_identifier_eprime_str = signal1_epochs.metadata.iloc[i].get('trial_identifier_eprime', trial_identifier_eprime_str)

            # Ensure signals are float64 for hilbert
            signal1_trial_data_avg_float64 = signal1_trial_data_avg.astype(np.float64)
            signal2_segment_for_plv_float64 = signal2_segment_for_plv.astype(np.float64)

            # Apply Hilbert transform and get phase for signal 2
            # Subtract mean before Hilbert for continuous signal to remove DC offset
            analytic_signal2: np.ndarray = hilbert(signal2_segment_for_plv_float64 - np.mean(signal2_segment_for_plv_float64)) # type: ignore
            phase_signal2_epoch = np.angle(analytic_signal2) # type: ignore

            for band_name, band_freqs in signal1_bands_config.items():
                # Filter signal1 into the current band
                signal1_filtered_band = mne.filter.filter_data(signal1_trial_data_avg_float64, signal1_epoch_sfreq,
                                                           l_freq=band_freqs[0], h_freq=band_freqs[1],
                                                           verbose=False, fir_design='firwin')
                # Apply Hilbert transform and get phase for the filtered signal1
                analytic_signal1_band: np.ndarray = hilbert(signal1_filtered_band) # type: ignore
                phase_signal1_epoch_band = np.angle(analytic_signal1_band) # type: ignore

                # Calculate PLV for this band and this epoch
                plv_val = _calculate_plv_segment(phase_signal1_epoch_band, phase_signal2_epoch, self.logger)
                if not np.isnan(plv_val):
                    all_trial_plv_results.append({
                        self.OUTPUT_COL_PARTICIPANT_ID: participant_id,
                        self.OUTPUT_COL_CONDITION: condition_name,
                        self.OUTPUT_COL_EPOCH_INDEX: i,
                        self.OUTPUT_COL_TRIAL_ID_EPRIME: trial_identifier_eprime_str,
                        self.OUTPUT_COL_MODALITY_PAIR: f"{signal1_name}-{signal2_name}",
                        self.OUTPUT_COL_SIGNAL1_BAND: band_name,
                        self.OUTPUT_COL_PLV_VALUE: plv_val
                    })

        self.logger.info(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): PLV calculation for P:{participant_id} ({signal1_name}-{signal2_name}) completed. Found {len(all_trial_plv_results)} PLV values.")
        return pd.DataFrame(all_trial_plv_results)