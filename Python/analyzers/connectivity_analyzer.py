import mne
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
                                               ) -> Optional[Union[mne.connectivity.SpectralConnectivity, List[mne.connectivity.SpectralConnectivity]]]:
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

        actual_picks = picks if picks is not None else mne.pick_types(epochs.info, eeg=True)
        if not actual_picks.size: # MNE pick_types returns an array
            self.logger.error("ConnectivityAnalyzer: No EEG channels found or selected. Cannot calculate connectivity.")
            return None

        self.logger.info(f"ConnectivityAnalyzer: Calculating spectral connectivity with method(s) '{method}', mode '{mode}'.")
        try:
            con = mne.connectivity.spectral_connectivity_epochs(
                epochs, method=method, mode=mode, indices=indices,
                sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax, faverage=faverage,
                tmin=tmin, tmax=tmax, mt_bandwidth=mt_bandwidth, mt_low_bias=mt_low_bias,
                picks=actual_picks, n_jobs=n_jobs, verbose=False
            )
            return con
        except Exception as e:
            self.logger.error(f"ConnectivityAnalyzer: Error calculating spectral connectivity: {e}", exc_info=True)
            return None
        
    def _get_aligned_continuous_segment(self, epoch_start_time_abs: float, epoch_duration_sec: float,
                                        continuous_signal_full: np.ndarray, continuous_signal_sfreq_orig: float,
                                        target_epoch_len_samples: int, target_epoch_sfreq: float,
                                        continuous_signal_name: str) -> Optional[np.ndarray]:
        """
        Helper: Extracts, resamples, and pads/truncates a continuous signal segment
        to match an epoch's timing and length.
        """
        if continuous_signal_full is None or continuous_signal_sfreq_orig is None:
            self.logger.debug(f"ConnectivityAnalyzer (Alignment): Continuous signal ({continuous_signal_name}) or its sfreq is None.")
            return None

        start_sample_orig = int(epoch_start_time_abs * continuous_signal_sfreq_orig)
        # Ensure end sample does not exceed signal length
        end_sample_orig = min(
            int((epoch_start_time_abs + epoch_duration_sec) * continuous_signal_sfreq_orig),
            len(continuous_signal_full)
        )

        if start_sample_orig < 0 or start_sample_orig >= end_sample_orig:
            self.logger.debug(f"ConnectivityAnalyzer (Alignment): Continuous signal segment indices out of bounds or invalid for {continuous_signal_name}. Epoch Start: {epoch_start_time_abs:.2f}s, Duration: {epoch_duration_sec:.2f}s. Indices: {start_sample_orig}-{end_sample_orig}, SigLen: {len(continuous_signal_full)}")
            return None

        continuous_epoch_raw = continuous_signal_full[start_sample_orig:end_sample_orig]
        if continuous_epoch_raw.size == 0:
            self.logger.debug(f"ConnectivityAnalyzer (Alignment): Extracted continuous segment for {continuous_signal_name} is empty.")
            return None

        # Resample this specific epoch to the target sampling frequency (e.g., EEG sfreq)
        # MNE resample needs float64
        resampled_segment = mne.filter.resample(continuous_epoch_raw.astype(np.float64),
                                                up=target_epoch_sfreq,
                                                down=continuous_signal_sfreq_orig, # type: ignore[arg-type]
                                                npad='auto', verbose=False) # type: ignore[arg-type]

        # Pad or truncate to match the target length
        if len(resampled_segment) > target_epoch_len_samples:
            final_segment = resampled_segment[:target_epoch_len_samples]
        elif len(resampled_segment) < target_epoch_len_samples:
            pad_width = target_epoch_len_samples - len(resampled_segment)
            final_segment = np.pad(resampled_segment, (0, pad_width), 'edge')
        else:
            final_segment = resampled_segment

        return final_segment

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

        all_trial_plv_results = []
        signal1_epoch_sfreq = signal1_epochs.info['sfreq']

        if not signal1_bands_config:
            self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Signal 1 bands configuration is empty for P:{participant_id}. Skipping PLV calculation for {signal1_name}-{signal2_name}.")
            return pd.DataFrame()

        for i, epoch_s1 in enumerate(signal1_epochs): # Iterate through each trial of signal1
            condition_code = epoch_s1.events[0,2]
            condition_name = "UnknownCondition"
            for name, code in signal1_epochs.event_id.items():
                if code == condition_code:
                    condition_name = name
                    break

            signal1_trial_data_multichannel = epoch_s1.get_data(picks=signal1_channels_to_average)
            if signal1_trial_data_multichannel.size == 0:
                self.logger.warning(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): Empty data for signal 1 in trial {i}, P:{participant_id}, cond {condition_name}. Skipping.")
                continue

            signal1_trial_data_avg = signal1_trial_data_multichannel.mean(axis=0).ravel() # Average selected channels
            target_epoch_len_samples = len(signal1_trial_data_avg)

            # Determine absolute start and end times of the EEG epoch
            event_sample_in_raw = epoch_s1.events[0,0]
            epoch_tmin_from_event = epoch_s1.tmin

            trial_identifier_eprime_str = self.DEFAULT_TRIAL_IDENTIFIER_EPRIME
            if signal1_epochs.metadata is not None and 'trial_identifier_eprime' in signal1_epochs.metadata.columns:
                trial_identifier_eprime_str = signal1_epochs.metadata.iloc[i].get('trial_identifier_eprime', trial_identifier_eprime_str)
            else:
                self.logger.debug(f"ConnectivityAnalyzer (Epoched vs Continuous PLV): 'trial_identifier_eprime' column not found in signal1_epochs metadata for P:{participant_id}, trial_idx {i}, cond {condition_name}.")

            trial_start_time_abs = (event_sample_in_raw / signal1_original_sfreq_for_event_timing) + epoch_tmin_from_event
            trial_duration_sec = target_epoch_len_samples / signal1_epoch_sfreq

            # Get aligned segment for signal2
            signal2_segment_for_plv = self._get_aligned_continuous_segment(
                trial_start_time_abs, trial_duration_sec,
                signal2_continuous_data, signal2_sfreq,
                target_epoch_len_samples, signal1_epoch_sfreq, signal2_name
            )

            if signal2_segment_for_plv is not None:
                # Ensure signals are float64 for hilbert
                signal1_trial_data_avg_float64 = signal1_trial_data_avg.astype(np.float64)
                signal2_segment_for_plv_float64 = signal2_segment_for_plv.astype(np.float64)

                # Apply Hilbert transform and get phase
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