import neurokit2 as nk
import pandas as pd
import os
import numpy as np
from typing import Optional, Tuple, Union, List, Any

# Module-level defaults for EDAPreprocessor
# Default parameters for the NeuroKit2 processing pipeline
# This is the 'method' argument for nk.eda_process, which controls nk.eda_clean's method.
DEFAULT_EDA_CLEANING_METHOD = "neurokit"
# Default column names from nk.eda_process output and for saved files
DEFAULT_PHASIC_COL_NAME = "EDA_Phasic"
DEFAULT_TONIC_COL_NAME = "EDA_Tonic"
DEFAULT_PHASIC_FILENAME_SUFFIX = "_phasic_eda.csv"
DEFAULT_TONIC_FILENAME_SUFFIX = "_tonic_eda.csv"
class EDAPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDAPreprocessor initialized.")
    def preprocess_eda(self,
                         eda_signal_raw: np.ndarray,
                         eda_sampling_rate: Union[int, float],
                         participant_id: str,
                         output_dir: str,
                         eda_cleaning_method_config: Optional[str] = None)
                         -> Tuple[Optional[str], Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Processes raw EDA signal to extract and save phasic and tonic components.
        Args:
            eda_signal_raw (np.ndarray): The raw EDA signal.
            eda_sampling_rate (Union[int, float]): The sampling rate of the EDA signal.
            participant_id (str): The participant ID.
            output_dir (str): Directory to save the output file.
            eda_cleaning_method_config (Optional[str]): Method for nk.eda_clean.
                                                        Defaults to DEFAULT_EDA_CLEANING_METHOD.
        Returns:
            tuple: (phasic_eda_path, tonic_eda_path, phasic_eda_array, tonic_eda_array)
                   Returns (None, None, None, None) if error.
        """
        if eda_signal_raw is None or eda_sampling_rate is None:
            self.logger.warning("EDAPreprocessor - Raw EDA signal or sampling rate not provided. Skipping.")
            return None, None, None, None
        
        if eda_sampling_rate <= 0:
            self.logger.error(f"EDAPreprocessor - Invalid EDA sampling rate: {eda_sampling_rate}. Skipping.")
            return None, None, None, None

        # Determine final EDA cleaning method
        final_cleaning_method = DEFAULT_EDA_CLEANING_METHOD # Start with default
        if eda_cleaning_method_config is not None: # If user provided something
            if isinstance(eda_cleaning_method_config, str) and eda_cleaning_method_config.strip():
                final_cleaning_method = eda_cleaning_method_config.strip()
            else:
                self.logger.warning(
                    f"EDAPreprocessor: Invalid value ('{eda_cleaning_method_config}') provided for 'eda_cleaning_method_config'. "
                    f"Expected a non-empty string. Using default: '{DEFAULT_EDA_CLEANING_METHOD}'."
                )
        # If eda_cleaning_method_config was None, final_cleaning_method remains the default.

        self.logger.info(f"EDAPreprocessor - Processing EDA for {participant_id} using cleaning method '{final_cleaning_method}'.")

        try:
            # Ensure eda_signal_raw is a 1D numpy array for neurokit2 processing
            processed_eda_signal: Optional[np.ndarray] = None

            if isinstance(eda_signal_raw, pd.Series):
                processed_eda_signal = eda_signal_raw.values
            elif isinstance(eda_signal_raw, (list, np.ndarray)):
                temp_array = np.array(eda_signal_raw)
                if temp_array.ndim > 1:
                    if temp_array.shape[0] == 1 or temp_array.shape[1] == 1:
                        processed_eda_signal = temp_array.ravel() # Flatten if it's a row or column vector
                    else:
                        self.logger.error("EDAPreprocessor - EDA signal is not 1D. Cannot process.")
                        return None, None, None, None
                else:
                    processed_eda_signal = temp_array # It's already a 1D numpy array

            if processed_eda_signal is None: # Should not happen with the above logic, but safety check
                 self.logger.error("EDAPreprocessor - Failed to prepare EDA signal for processing.")
                 return None, None, None, None

            # Use the defined default cleaning method for nk.eda_process
            eda_signals, _ = nk.eda_process(processed_eda_signal, sampling_rate=int(eda_sampling_rate), method=final_cleaning_method)
            
            phasic_eda: np.ndarray = eda_signals[DEFAULT_PHASIC_COL_NAME].values
            tonic_eda: np.ndarray = eda_signals[DEFAULT_TONIC_COL_NAME].values
            self.logger.debug("EDAPreprocessor - EDA signal decomposed, phasic and tonic components extracted.")

            phasic_eda_path = os.path.join(output_dir, f"{participant_id}{DEFAULT_PHASIC_FILENAME_SUFFIX}")
            pd.DataFrame(phasic_eda, columns=[DEFAULT_PHASIC_COL_NAME]).to_csv(phasic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Phasic EDA saved to {phasic_eda_path}")

            tonic_eda_path = os.path.join(output_dir, f"{participant_id}{DEFAULT_TONIC_FILENAME_SUFFIX}")
            pd.DataFrame(tonic_eda, columns=[DEFAULT_TONIC_COL_NAME]).to_csv(tonic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Tonic EDA saved to {tonic_eda_path}")

            return phasic_eda_path, tonic_eda_path, phasic_eda, tonic_eda
        except Exception as e:
            self.logger.error(f"EDAPreprocessor - Error processing EDA for {participant_id}: {e}", exc_info=True)
            return None, None, None, None