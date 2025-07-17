import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Tuple, Optional, Dict, Any

class EDASCRProcessor:
    # Default parameters for SCR processing
    DEFAULT_SCR_PEAK_METHOD = "neurokit"
    DEFAULT_SCR_AMPLITUDE_MIN = 0.01

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDASCRProcessor initialized.")

    def process_phasic_to_scr_features(self,
                                       phasic_eda_signal: np.ndarray,
                                       eda_sampling_rate: int,
                                       participant_id: str,
                                       output_dir: str,
                                       scr_peak_method: str = DEFAULT_SCR_PEAK_METHOD,
                                       scr_amplitude_min: float = DEFAULT_SCR_AMPLITUDE_MIN
                                       ) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """
        Detects all SCRs from a phasic EDA signal and extracts their features.

        Args:
            phasic_eda_signal (np.ndarray): The full preprocessed phasic EDA signal.
            eda_sampling_rate (int): Sampling rate of the EDA signal.
            participant_id (str): Participant ID for naming output files.
            output_dir (str): Directory to save the SCR features file.
            scr_peak_method (str): Method for SCR peak detection. Defaults to EDASCRProcessor.DEFAULT_SCR_PEAK_METHOD.
            scr_amplitude_min (float): Minimum amplitude threshold for SCRs. Defaults to EDASCRProcessor.DEFAULT_SCR_AMPLITUDE_MIN.

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
        if not isinstance(eda_sampling_rate, int):
            self.logger.error(f"EDASCRProcessor - Invalid type for EDA sampling rate: {type(eda_sampling_rate)}. Expected int. Skipping.")
            return None, None
        if eda_sampling_rate <= 0:
            self.logger.error(f"EDASCRProcessor - Invalid EDA sampling rate: {eda_sampling_rate}. Skipping.")
            return None, None
            
        self.logger.info(f"EDASCRProcessor - P:{participant_id}: Detecting SCRs and extracting features.")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: Created output directory {output_dir}")
            except Exception as e_mkdir:
                self.logger.error(f"EDASCRProcessor - P:{participant_id}: Failed to create output directory {output_dir}: {e_mkdir}", exc_info=True)
                return None, None # Cannot save files


        try:
            if not isinstance(phasic_eda_signal, np.ndarray):
                self.logger.error(f"EDASCRProcessor - Expected 'phasic_eda_signal' to be a numpy array, but got {type(phasic_eda_signal)}. Cannot proceed.")
                return None, None 

            _, info = nk.eda_peaks(
                phasic_eda_signal,
                sampling_rate=eda_sampling_rate,
                method=scr_peak_method,
                amplitude_min=scr_amplitude_min
            )

            # Explicitly extract features based on docstring and standard NeuroKit2 keys
            scr_features_data = {}
            if info.get('SCR_Onsets') is not None:
                scr_features_data['SCR_Onset_Time'] = np.array(info['SCR_Onsets']) / eda_sampling_rate
            if info.get('SCR_Peaks') is not None:
                scr_features_data['SCR_Peak_Time'] = np.array(info['SCR_Peaks']) / eda_sampling_rate
            if info.get('SCR_Amplitude') is not None:
                scr_features_data['SCR_Amplitude'] = np.array(info['SCR_Amplitude'])
            if info.get('SCR_RiseTime') is not None: # NeuroKit2 typically provides this in seconds
                scr_features_data['SCR_RiseTime'] = np.array(info['SCR_RiseTime'])
            if info.get('SCR_RecoveryTime') is not None: # NeuroKit2 typically provides this in seconds
                scr_features_data['SCR_RecoveryTime'] = np.array(info['SCR_RecoveryTime'])

            # If scr_features_data remains empty, it means no relevant keys were found in info,
            # or their values were None. pd.DataFrame({}) will create an empty DataFrame.
            # If only some keys are present, DataFrame will be created with available columns.
            # This assumes that if keys are present, their corresponding value arrays (lists from info)
            # will have consistent lengths for all detected SCRs.

            scr_features_df = pd.DataFrame(scr_features_data)

            if scr_features_df.empty:
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: No SCRs detected with the given parameters.")
            else:
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: Detected {len(scr_features_df)} SCRs.")

            scr_features_path = os.path.join(output_dir, f"{participant_id}_eda_scr_features.csv")
            try:
                scr_features_df.to_csv(scr_features_path, index=False)
                self.logger.info(f"EDASCRProcessor - P:{participant_id}: SCR features saved to {scr_features_path}")
            except Exception as e_save_scr:
                self.logger.error(f"EDASCRProcessor - P:{participant_id}: Failed to save SCR features: {e_save_scr}", exc_info=True)
                scr_features_path = None # Indicate failure to save
            
            return scr_features_path, scr_features_df
            
        except Exception as e:
            self.logger.error(f"EDASCRProcessor - P:{participant_id}: Error during SCR feature extraction: {e}", exc_info=True)
            return None, None