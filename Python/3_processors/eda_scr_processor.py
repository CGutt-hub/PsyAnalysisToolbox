import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Tuple, Optional, Dict, Any

class EDASCRProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDASCRProcessor initialized.")

    def process_phasic_to_scr_features(self,
                                       phasic_eda_signal: np.ndarray,
                                       eda_sampling_rate: float,
                                       participant_id: str,
                                       output_dir: str,
                                       scr_peak_method: str = "neurokit",
                                       scr_amplitude_min: float = 0.01
                                       ) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """
        Detects all SCRs from a phasic EDA signal and extracts their features.

        Args:
            phasic_eda_signal (np.ndarray): The full preprocessed phasic EDA signal.
            eda_sampling_rate (float): Sampling rate of the EDA signal.
            participant_id (str): Participant ID for naming output files.
            output_dir (str): Directory to save the SCR features file.
            scr_peak_method (str): Method for SCR peak detection in NeuroKit2.
            scr_amplitude_min (float): Minimum amplitude threshold for detecting SCR peaks.

        Returns:
            Tuple containing:
                - scr_features_path (Optional[str]): Path to the saved SCR features CSV file.
                - scr_features_df (Optional[pd.DataFrame]): DataFrame containing features of detected SCRs.
                                                            Columns include: 'SCR_Onset_Time', 'SCR_Peak_Time',
                                                            'SCR_Amplitude', 'SCR_RiseTime', 'SCR_RecoveryTime'.
                                                            Times are in seconds from the start of the signal.
        """
        if phasic_eda_signal is None or eda_sampling_rate is None:
            self.logger.warning("EDASCRProcessor - Phasic EDA signal or sampling rate not provided. Skipping SCR feature extraction.")
            return None, None
        if eda_sampling_rate <= 0:
            self.logger.error(f"EDASCRProcessor - Invalid EDA sampling rate: {eda_sampling_rate}. Skipping.")
            return None, None

        self.logger.info(f"EDASCRProcessor - P:{participant_id}: Detecting SCRs and extracting features.")

        try:
            _, info = nk.eda_peaks(
                phasic_eda_signal,
                sampling_rate=eda_sampling_rate,
                method=scr_peak_method,
                amplitude_min=scr_amplitude_min
            )

            # Create a DataFrame from the 'info' dictionary
            # Convert sample indices to times in seconds
            scr_features_data = {}
            for key, values in info.items():
                if key in ['SCR_Onsets', 'SCR_Peaks', 'SCR_RecoveryTime_Onsets']: # These are sample indices
                    scr_features_data[f"{key}_Time"] = np.array(values) / eda_sampling_rate
                elif key in ['SCR_Amplitude', 'SCR_RiseTime', 'SCR_RecoveryTime']: # These are already values or durations in seconds
                     scr_features_data[key] = np.array(values)
                # We can add more features from 'info' if needed

            scr_features_df = pd.DataFrame(scr_features_data)

            if scr_features_df.empty:
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: No SCRs detected with the given parameters.")
            else:
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: Detected {len(scr_features_df)} SCRs.")

            scr_features_path = os.path.join(output_dir, f"{participant_id}_eda_scr_features.csv")
            scr_features_df.to_csv(scr_features_path, index=False)
            self.logger.info(f"EDASCRProcessor - P:{participant_id}: SCR features saved to {scr_features_path}")

            return scr_features_path, scr_features_df

        except Exception as e:
            self.logger.error(f"EDASCRProcessor - P:{participant_id}: Error during SCR feature extraction: {e}", exc_info=True)
            return None, None