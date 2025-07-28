import neurokit2 as nk
import pandas as pd
import os
import numpy as np
import logging
from typing import Optional, Tuple, Union, List, Any, Dict
class EDAPreprocessor:
    """
    Universal EDA preprocessing module for phasic/tonic decomposition using NeuroKit2.
    - Accepts a config dict with required and optional keys.
    - Fills in missing keys with class-level defaults.
    - Raises clear errors for missing required keys.
    - Usable in any project (no project-specific assumptions).

    Required config keys:
        - 'eda_cleaning_method': str (e.g., 'neurokit', 'biosppy', etc.)
    Optional config keys (with defaults):
        - None currently, but can be extended.
    """
    # Class-level defaults
    # Default parameters for the NeuroKit2 processing pipeline
    # This is the 'method' argument for nk.eda_process, which controls nk.eda_clean's method.
    DEFAULT_EDA_CLEANING_METHOD = "neurokit"
    # Default column names from nk.eda_process output and for saved files
    DEFAULT_PHASIC_COL_NAME = "EDA_Phasic"
    DEFAULT_TONIC_COL_NAME = "EDA_Tonic"
    DEFAULT_PHASIC_FILENAME_SUFFIX = "_phasic_eda.csv"
    DEFAULT_TONIC_FILENAME_SUFFIX = "_tonic_eda.csv"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("EDAPreprocessor initialized.")

    def preprocess_eda(self,
                         eda_signal_raw: np.ndarray,
                         eda_sampling_rate: Union[int, float],
                         participant_id: str,
                         output_dir: str,
                         eda_cleaning_method_config: Optional[str] = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
 Processes raw EDA signal to extract and save phasic and tonic components.
        Args:
            eda_signal_raw (Union[np.ndarray, pd.Series]): The raw EDA signal.
            eda_sampling_rate (Union[int, float]): The sampling rate of the EDA signal.
            participant_id (str): The participant ID.
            output_dir (str): Directory to save the output file.
            eda_cleaning_method_config (Optional[str]): Method for nk.eda_clean.
                                                        Defaults to DEFAULT_EDA_CLEANING_METHOD.
        Returns:
            A tuple containing two DataFrames (phasic_df, tonic_df), or None if an error occurs.
        """
        if eda_signal_raw is None or eda_sampling_rate is None:
            self.logger.warning("EDAPreprocessor - Raw EDA signal or sampling rate not provided. Skipping.")
            return None

        if not isinstance(eda_sampling_rate, (int, float)):
            self.logger.error(f"EDAPreprocessor - Invalid type for EDA sampling rate: {type(eda_sampling_rate)}. Expected int or float. Skipping.")
            return None
        if eda_sampling_rate <= 0:
            self.logger.error(f"EDAPreprocessor - Invalid EDA sampling rate: {eda_sampling_rate}. Skipping.")
            return None

        # Determine final EDA cleaning method
        final_cleaning_method = self.DEFAULT_EDA_CLEANING_METHOD # Start with default
        if eda_cleaning_method_config is not None: # If user provided something
            if isinstance(eda_cleaning_method_config, str) and eda_cleaning_method_config.strip():
                final_cleaning_method = eda_cleaning_method_config.strip()
            else:
                self.logger.warning(
                    f"EDAPreprocessor: Invalid value ('{eda_cleaning_method_config}') provided for 'eda_cleaning_method_config'. "
                    f"Expected a non-empty string. Using default: '{self.DEFAULT_EDA_CLEANING_METHOD}'."
                )
        # If eda_cleaning_method_config was None, final_cleaning_method remains the default.

        self.logger.info(f"EDAPreprocessor - Processing EDA for {participant_id} using cleaning method '{final_cleaning_method}'.")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"EDAPreprocessor - Created output directory {output_dir} for P:{participant_id}")
            except Exception as e_mkdir:
                self.logger.error(f"EDAPreprocessor - Failed to create output directory {output_dir} for P:{participant_id}: {e_mkdir}", exc_info=True)
                return None # Cannot save files

        try:
            # Ensure eda_signal_raw is a 1D numpy array for neurokit2 processing
            processed_eda_signal: Optional[np.ndarray] = None # Declare type for clarity

            if isinstance(eda_signal_raw, pd.Series):
                processed_eda_signal = np.asarray(eda_signal_raw.values) # Ensure it's a numpy array
            elif isinstance(eda_signal_raw, (list, np.ndarray)):
                temp_array = np.array(eda_signal_raw)
                if temp_array.ndim > 1:
                    if temp_array.shape[0] == 1 or temp_array.shape[1] == 1:
                        processed_eda_signal = temp_array.ravel() # Flatten if it's a row or column vector
                    else:
                        self.logger.error("EDAPreprocessor - EDA signal is not 1D. Cannot process.")
                        return None
                else:
                    processed_eda_signal = temp_array # It's already a 1D numpy array

            if processed_eda_signal is None: # Should not happen with the above logic, but safety check
                 self.logger.error("EDAPreprocessor - Failed to prepare EDA signal for processing.")
                 return None

            # Ensure processed_eda_signal is treated as a numpy array for type hinting/safety
            if not isinstance(processed_eda_signal, np.ndarray):
                 self.logger.error("EDAPreprocessor - Processed EDA signal is not a numpy array after preparation. Cannot proceed.")
                 return None
 
            # Now we are confident processed_eda_signal is a np.ndarray

            # Use the defined default cleaning method for nk.eda_process
            eda_signals, _ = nk.eda_process(processed_eda_signal, sampling_rate=int(eda_sampling_rate), method=final_cleaning_method) # type: ignore[arg-type] # processed_eda_signal is np.ndarray, which is ArrayLike
            
            phasic_eda: np.ndarray = np.asarray(eda_signals[self.DEFAULT_PHASIC_COL_NAME].values)
            tonic_eda: np.ndarray = np.asarray(eda_signals[self.DEFAULT_TONIC_COL_NAME].values) # type: ignore
            eda_times = np.arange(len(processed_eda_signal)) / eda_sampling_rate # Create a times array (in seconds)
            self.logger.debug("EDAPreprocessor - EDA signal decomposed, phasic and tonic components extracted.")
 
            phasic_eda_path = os.path.join(output_dir, f"{participant_id}{self.DEFAULT_PHASIC_FILENAME_SUFFIX}")
            pd.DataFrame(phasic_eda, columns=[self.DEFAULT_PHASIC_COL_NAME]).to_csv(phasic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Phasic EDA saved to {phasic_eda_path}")

            tonic_eda_path = os.path.join(output_dir, f"{participant_id}{self.DEFAULT_TONIC_FILENAME_SUFFIX}")
            pd.DataFrame(tonic_eda, columns=[self.DEFAULT_TONIC_COL_NAME]).to_csv(tonic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Tonic EDA saved to {tonic_eda_path}")

            # Create DataFrames for return
            phasic_df = pd.DataFrame({
                'time_sec': eda_times,
                'EDA_Phasic': phasic_eda,
                'participant_id': participant_id # Add participant ID
            })
            tonic_df = pd.DataFrame({
                'time_sec': eda_times,
                'EDA_Tonic': tonic_eda, # type: ignore
                'participant_id': participant_id
            })

            self.logger.info(f"EDAPreprocessor - Preprocessed EDA returned as a two DataFrames, with phasuic components, shape {phasic_df.shape} and tonic components, shape {tonic_df.shape}.")

            return phasic_df, tonic_df


        except Exception as e:
            self.logger.error(f"EDAPreprocessor - Error processing EDA for {participant_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def default_config():
        """Return a default config dict for typical EDA preprocessing."""
        return {
            'eda_cleaning_method': EDAPreprocessor.DEFAULT_EDA_CLEANING_METHOD
        }

    def _validate_and_resolve_config(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validates and fills defaults for the EDA configuration. Returns None on validation failure."""
        if not isinstance(config, dict):
            cfg = dict(config)
        else:
            cfg = config.copy()
        # Required key
        if 'eda_cleaning_method' not in cfg or not isinstance(cfg['eda_cleaning_method'], str):
            self.logger.error("EDAPreprocessor - Missing or invalid required config key: 'eda_cleaning_method'.")
            return None
        return cfg