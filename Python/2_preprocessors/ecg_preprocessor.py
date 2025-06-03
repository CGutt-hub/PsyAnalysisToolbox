import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Tuple, Optional # Added for more specific return type hinting

class ECGPreprocessor:
    # Default parameters
    DEFAULT_RPEAK_DETECTION_METHOD = "neurokit" # Default for nk.ecg_peaks
    DEFAULT_RPEAK_FILENAME_SUFFIX = "_ecg_rpeak_times.csv"
    DEFAULT_RPEAK_COLUMN_NAME = "R_Peak_Time_s"

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ECGPreprocessor initialized.")

    def preprocess_ecg(self,
                         ecg_signal: np.ndarray,
                         ecg_sfreq: float,
                         participant_id: str,
                         preproc_results_dir: str,
                         ecg_rpeak_method_config: str = DEFAULT_RPEAK_DETECTION_METHOD) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Preprocesses ECG data to detect R-peaks and calculate NN intervals.

        Args:
            ecg_signal (np.ndarray): The raw ECG signal.
            ecg_sfreq (float): The sampling frequency of the ECG signal.
            participant_id (str): The ID of the participant.
            preproc_results_dir (str): Directory to save intermediate results.
            ecg_rpeak_method_config (str): The method for R-peak detection.
                                           Defaults to ECGPreprocessor.DEFAULT_RPEAK_DETECTION_METHOD.

        Returns:
            tuple: (rpeak_times_path, rpeaks_samples_array)
                   Returns (None, None) if preprocessing fails.
        """
        if ecg_signal is None or ecg_sfreq is None:
            self.logger.warning("ECGPreprocessor - No ECG signal or sampling frequency provided. Skipping preprocessing.")
            return None, None
        if not ecg_rpeak_method_config:
            self.logger.warning("ECGPreprocessor - ECG R-peak detection method not provided. Skipping.")
            return None, None

        self.logger.info(f"ECGPreprocessor - Starting preprocessing for {participant_id}.")

        try:
            # 2. Find R-peaks
            # Use NeuroKit2's ecg_peaks function
            self.logger.info("ECGPreprocessor - Detecting R-peaks...")
            # The `method` parameter uses the specified algorithm from the passed config
            signals, info = nk.ecg_peaks(ecg_signal, sampling_rate=ecg_sfreq, method=ecg_rpeak_method_config)

            rpeaks = info["ECG_R_Peaks"]
            self.logger.info(f"ECGPreprocessor - Found {len(rpeaks)} R-peaks.")

            if len(rpeaks) < 2:
                self.logger.warning("ECGPreprocessor - Less than 2 R-peaks found. R-peak processing might be incomplete for HRV.")
                # Let's return what we have for R-peaks, but downstream will handle NNI issues.
            # Get R-peak times in seconds relative to the start of the signal
            rpeak_times_s = rpeaks / ecg_sfreq

            # 4. Save results
            rpeak_times_df = pd.DataFrame({self.DEFAULT_RPEAK_COLUMN_NAME: rpeak_times_s})
            rpeak_times_path = os.path.join(preproc_results_dir, f'{participant_id}{self.DEFAULT_RPEAK_FILENAME_SUFFIX}')
            rpeak_times_df.to_csv(rpeak_times_path, index=False)

            self.logger.info(f"ECGPreprocessor - R-peak times saved to {rpeak_times_path}")
            self.logger.info(f"ECGPreprocessor - Preprocessing completed for {participant_id}.")
            return rpeak_times_path, rpeaks
        except Exception as e:
            self.logger.error(f"ECGPreprocessor - Error during preprocessing for {participant_id}: {e}", exc_info=True)
            return None, None