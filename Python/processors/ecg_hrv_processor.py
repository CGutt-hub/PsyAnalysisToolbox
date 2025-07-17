import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Tuple, Optional, cast

class ECGHRVProcessor:
    # Default parameters for HRV processing
    DEFAULT_TARGET_SFREQ_CONTINUOUS_HRV = 4.0 # Hz

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ECGHRVProcessor initialized.")

    def process_rpeaks_to_hrv(self,
                              rpeaks_samples: np.ndarray,
                              original_sfreq: float,
                              participant_id: str,
                              output_dir: str,
                              target_sfreq_continuous_hrv: float = DEFAULT_TARGET_SFREQ_CONTINUOUS_HRV) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
       Processes R-peak data to calculate NN-intervals, overall RMSSD, and a continuous HRV signal.

       Args:
        rpeaks_samples (np.ndarray): Array of R-peak sample indices.

           original_sfreq (float): Sampling frequency of the signal from which R-peaks were derived.
           participant_id (str): Participant ID for naming output files.
           output_dir (str): Directory to save processed files (NN-intervals, continuous HRV).
           target_sfreq_continuous_hrv (float): Target sfreq for continuous HRV. Defaults to ECGHRVProcessor.DEFAULT_TARGET_SFREQ_CONTINUOUS_HRV.

   Returns:
            Tuple containing:
                - nn_intervals_path (Optional[str]): Path to the saved NN-intervals CSV file.
            - nn_intervals_ms (Optional[np.ndarray]): Array of NN-intervals in milliseconds.
                - continuous_hrv_signal (Optional[np.ndarray]): The continuous HRV signal (interpolated NNIs in ms).
            - continuous_hrv_time_vector (Optional[np.ndarray]): Time vector for the continuous HRV signal.
            continuous_hrv_signal_path (Optional[str]): Path to the saved continuous HRV signal CSV file."""
        if rpeaks_samples is None or len(rpeaks_samples) < 2:
            self.logger.warning(f"ECGHRVProcessor - P:{participant_id}: Not enough R-peaks ({len(rpeaks_samples) if rpeaks_samples is not None else 0}) provided. Skipping HRV processing.")
            return None, None, None, None, None
        if not isinstance(original_sfreq, (int, float)) or original_sfreq <= 0:
            self.logger.error(f"ECGHRVProcessor - P:{participant_id}: Invalid original_sfreq ({original_sfreq}). Skipping.")
            return None, None, None, None, None # type: ignore
        if target_sfreq_continuous_hrv <= 0:
            self.logger.error(f"ECGHRVProcessor - P:{participant_id}: Invalid target_sfreq_continuous_hrv ({target_sfreq_continuous_hrv}). Skipping.")
            return None, None, None, None, None # type: ignore


        self.logger.info(f"ECGHRVProcessor - P:{participant_id}: Processing R-peaks to HRV features.")

        # 1. Calculate NN intervals
        nn_intervals_ms = np.diff(rpeaks_samples) / original_sfreq * 1000  # ms
        nn_intervals_s = nn_intervals_ms / 1000  # s

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"ECGHRVProcessor - P:{participant_id}: Created output directory {output_dir}")
            except Exception as e_mkdir:
                self.logger.error(f"ECGHRVProcessor - P:{participant_id}: Failed to create output directory {output_dir}: {e_mkdir}", exc_info=True) # type: ignore
                return None, None, None, None, None # type: ignore

        # Save NN intervals
        nn_intervals_df = pd.DataFrame({'NN_Interval_ms': nn_intervals_ms, 'NN_Interval_s': nn_intervals_s})
        nn_intervals_path = os.path.join(output_dir, f'{participant_id}_ecg_nn_intervals.csv')
        try:
            nn_intervals_df.to_csv(nn_intervals_path, index=False)
            self.logger.info(f"ECGHRVProcessor - P:{participant_id}: NN intervals saved to {nn_intervals_path}")
        except Exception as e_save_nni:
            self.logger.error(f"ECGHRVProcessor - P:{participant_id}: Failed to save NN intervals: {e_save_nni}", exc_info=True)
            nn_intervals_path = None # Indicate failure

        # 3. Generate continuous HRV signal
        continuous_hrv_signal = None
        continuous_hrv_time_vector = None
        continuous_hrv_signal_path = None

        if len(nn_intervals_ms) >= 2: # Need at least one NNI, which means at least 2 R-peaks
            rpeak_times_sec = rpeaks_samples / original_sfreq
            nn_interval_times_sec = rpeak_times_sec[1:] # Times corresponding to the end of each NNI

            if len(nn_interval_times_sec) == len(nn_intervals_ms) and \
               not np.any(np.isnan(nn_intervals_ms)) and not np.any(np.isinf(nn_intervals_ms)) and \
               not np.any(np.isnan(nn_interval_times_sec)) and not np.any(np.isinf(nn_interval_times_sec)) and \
               (len(nn_interval_times_sec) <= 1 or np.all(np.diff(nn_interval_times_sec) > 0)):

                # Since the outer condition is len(nn_intervals_ms) >= 2, num_nni_points is also >= 2.
                num_nni_points = len(nn_intervals_ms) # This will be >= 2
                interp_kind = 'linear' # Default for 2 points
                if num_nni_points >= 3: interp_kind = 'quadratic'
                if num_nni_points >= 4: interp_kind = 'cubic'
                
                self.logger.debug(f"ECGHRVProcessor - P:{participant_id}: Using '{interp_kind}' interpolation for {num_nni_points} NNI points for continuous HRV.")
                interp_func = interp1d(nn_interval_times_sec, nn_intervals_ms, kind=interp_kind, fill_value="extrapolate", bounds_error=False) # type: ignore

                min_time_interp = nn_interval_times_sec[0]
                max_time_interp = nn_interval_times_sec[-1]

                if max_time_interp > min_time_interp: # Ensure a valid time range for arange
                    continuous_hrv_time_vector = np.arange(min_time_interp, max_time_interp, 1.0 / target_sfreq_continuous_hrv)
                    if len(continuous_hrv_time_vector) > 0:
                        continuous_hrv_signal = interp_func(continuous_hrv_time_vector)
                        self.logger.info(f"ECGHRVProcessor - P:{participant_id}: Generated continuous HRV signal (NNIs in ms) at {target_sfreq_continuous_hrv} Hz.")

                        # Save continuous HRV signal
                        continuous_hrv_df = pd.DataFrame({'Time_s': continuous_hrv_time_vector, 'Interpolated_NNI_ms': continuous_hrv_signal})
                        continuous_hrv_signal_path = os.path.join(output_dir, f'{participant_id}_continuous_hrv_signal.csv')
                        try:
                            continuous_hrv_df.to_csv(continuous_hrv_signal_path, index=False)
                            self.logger.info(f"ECGHRVProcessor - P:{participant_id}: Continuous HRV signal saved to {continuous_hrv_signal_path}")
                        except Exception as e_save_cont_hrv:
                            self.logger.error(f"ECGHRVProcessor - P:{participant_id}: Failed to save continuous HRV signal: {e_save_cont_hrv}", exc_info=True)
                            continuous_hrv_signal_path = None
                    else:
                        self.logger.warning(f"ECGHRVProcessor - P:{participant_id}: Interpolated time vector for continuous HRV is empty. Check duration and target sfreq.")
                else:
                    self.logger.warning(f"ECGHRVProcessor - P:{participant_id}: Max time not greater than min time for NNI interpolation (min_time={min_time_interp}, max_time={max_time_interp}).")
            else:
                self.logger.warning(f"ECGHRVProcessor - P:{participant_id}: Issues with NN interval data (NaNs, Infs, non-monotonic times, or mismatch lengths). Cannot generate continuous HRV.")
        else: # len(nn_intervals_ms) < 2
            self.logger.warning(f"ECGHRVProcessor - P:{participant_id}: Not enough NN intervals ({len(nn_intervals_ms)}) to generate continuous HRV signal.")
        
        # Return all computed values and paths as per the function signature
        return nn_intervals_path, nn_intervals_ms, continuous_hrv_signal, continuous_hrv_time_vector, continuous_hrv_signal_path