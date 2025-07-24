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

    def analyze_scr(self,
                    phasic_eda_df: pd.DataFrame,
                    eda_sampling_rate: int,
                    scr_peak_method: str = DEFAULT_SCR_PEAK_METHOD,
                    scr_amplitude_min: float = DEFAULT_SCR_AMPLITUDE_MIN
                    ) -> Optional[pd.DataFrame]:
        """
        Detects all SCRs from a phasic EDA signal DataFrame and extracts their features.

        Args:
            phasic_eda_df (pd.DataFrame): DataFrame containing the preprocessed phasic EDA signal.
                                          Must contain a column named 'EDA_Phasic'.
            eda_sampling_rate (int): Sampling rate of the EDA signal.
            scr_peak_method (str): Method for SCR peak detection. Defaults to EDASCRProcessor.DEFAULT_SCR_PEAK_METHOD.
            scr_amplitude_min (float): Minimum amplitude threshold for SCRs. Defaults to EDASCRProcessor.DEFAULT_SCR_AMPLITUDE_MIN.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing features of detected SCRs, or None on error.
                                    Columns include: 'SCR_Onset_Time', 'SCR_Peak_Time',
                                    'SCR_Amplitude', 'SCR_RiseTime', 'SCR_RecoveryTime'.
                                    Times are in seconds from the start of the signal.
        """
        if phasic_eda_df is None or phasic_eda_df.empty or 'EDA_Phasic' not in phasic_eda_df.columns:
            self.logger.warning("EDASCRProcessor - Phasic EDA DataFrame is invalid or missing 'EDA_Phasic' column. Skipping SCR feature extraction.")
            return None
        if not isinstance(eda_sampling_rate, int) or eda_sampling_rate <= 0:
            self.logger.error(f"EDASCRProcessor - Invalid EDA sampling rate: {eda_sampling_rate}. Expected a positive integer. Skipping.")
            return None

        phasic_eda_signal = phasic_eda_df['EDA_Phasic'].to_numpy()
            
        self.logger.info(f"EDASCRProcessor - Detecting SCRs and extracting features.")

        try:
            if not isinstance(phasic_eda_signal, np.ndarray):
                self.logger.error(f"EDASCRProcessor - Expected 'phasic_eda_signal' to be a numpy array, but got {type(phasic_eda_signal)}. Cannot proceed.")
                return None 

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

            scr_features_df = pd.DataFrame(scr_features_data)

            if scr_features_df.empty:
                self.logger.info(f"EDASCRProcessor - No SCRs detected with the given parameters.")
            else:
                self.logger.info(f"EDASCRProcessor - Detected {len(scr_features_df)} SCRs.")
            
            return scr_features_df
            
        except Exception as e:
            self.logger.error(f"EDASCRProcessor - Error during SCR feature extraction: {e}", exc_info=True)
            return None