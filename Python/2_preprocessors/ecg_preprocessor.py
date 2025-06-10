import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Tuple, Optional # Added for more specific return type hinting

# Module-level defaults for ecg_preprocessor names
DEFAULT_RPEAK_DETECTION_METHOD = "neurokit" # Default for nk.ecg_peaks
DEFAULT_RPEAK_FILENAME_SUFFIX = "_ecg_rpeak_times.csv"
DEFAULT_RPEAK_COLUMN_NAME = "R_Peak_Time_s"

class ECGPreprocessor:

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ECGPreprocessor initialized.")

    def preprocess_ecg(self,
                         ecg_signal: np.ndarray,
                         ecg_sfreq: float,
                         participant_id: str,
                         output_dir: str,
                         ecg_rpeak_method_config: Optional[str] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Preprocesses ECG data to detect R-peaks.

        Args:
            ecg_signal (np.ndarray): The raw ECG signal.
            ecg_sfreq (float): The sampling frequency of the ECG signal.
            participant_id (str): The ID of the participant.
            output_dir (str): Directory to save intermediate results.
            ecg_rpeak_method_config (Optional[str]): The method for R-peak detection.
                                           If None, defaults to ECGPreprocessor.DEFAULT_RPEAK_DETECTION_METHOD.
 
            tuple: (rpeak_times_path, rpeaks_samples_array)
                   Returns (None, None) if preprocessing fails.
        """
        if ecg_signal is None or ecg_sfreq is None:
            self.logger.warning("ECGPreprocessor - No ECG signal or sampling frequency provided. Skipping preprocessing.")
            return None, None

        # Determine final R-peak detection method
        final_rpeak_method = DEFAULT_RPEAK_DETECTION_METHOD # Start with default
        if ecg_rpeak_method_config is not None: # If user provided something
            if isinstance(ecg_rpeak_method_config, str) and ecg_rpeak_method_config.strip():
                final_rpeak_method = ecg_rpeak_method_config.strip()
            else:
                self.logger.warning(
                    f"ECGPreprocessor: Invalid value ('{ecg_rpeak_method_config}') provided for 'ecg_rpeak_method_config'. "
                    f"Expected a non-empty string. Using default: '{DEFAULT_RPEAK_DETECTION_METHOD}'."
                )
        # If ecg_rpeak_method_config was None, final_rpeak_method remains the default.

        self.logger.info(f"ECGPreprocessor - Starting preprocessing for {participant_id} using R-peak method '{final_rpeak_method}'.")

        try:
            # 2. Find R-peaks
            # Use NeuroKit2's ecg_peaks function
            self.logger.info(f"ECGPreprocessor - Detecting R-peaks with method: {final_rpeak_method}...")
            # The `method` parameter uses the specified algorithm from the passed config
            signals, info = nk.ecg_peaks(ecg_signal, sampling_rate=int(ecg_sfreq), method=final_rpeak_method)
            
            # Ensure the expected output key exists and is the correct type
            rpeaks_raw = info.get("ECG_R_Peaks")
            if not isinstance(rpeaks_raw, np.ndarray):
                 self.logger.error("ECGPreprocessor - Expected 'ECG_R_Peaks' as numpy array from neurokit2, but got something else. Cannot proceed.")
                 return None, None

            # Now that we've checked, we can confidently assign the checked variable to the typed variable
            rpeaks: np.ndarray = rpeaks_raw # Assign the checked variable
            self.logger.info(f"ECGPreprocessor - Found {len(rpeaks)} R-peaks.")

            if len(rpeaks) < 2:
                self.logger.warning("ECGPreprocessor - Less than 2 R-peaks found. R-peak processing might be incomplete for HRV.", stacklevel=2)
                # Let's return what we have for R-peaks, but downstream will handle NNI issues.
            # Get R-peak times in seconds relative to the start of the signal
            rpeak_times_s = rpeaks / ecg_sfreq

            # 4. Save results
            rpeak_times_df = pd.DataFrame({DEFAULT_RPEAK_COLUMN_NAME: rpeak_times_s})
            rpeak_times_path: str = os.path.join(output_dir, f'{participant_id}{DEFAULT_RPEAK_FILENAME_SUFFIX}')
            rpeak_times_df.to_csv(rpeak_times_path, index=False)
            
            self.logger.info(f"ECGPreprocessor - R-peak times saved to {rpeak_times_path}")
            self.logger.info(f"ECGPreprocessor - Preprocessing completed for {participant_id}.")
            return rpeak_times_path, rpeaks
        except Exception as e:
            self.logger.error(f"ECGPreprocessor - Error during preprocessing for {participant_id}: {e}", exc_info=True)
            return None, None