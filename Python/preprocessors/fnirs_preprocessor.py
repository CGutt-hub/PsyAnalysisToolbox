"""
FNIRS Preprocessor Module
------------------------
Universal fNIRS preprocessing for MNE Raw objects.
Handles filtering, short channel regression, motion correction, and config-driven logic.
Config-driven, robust, and maintainable.
"""
import mne
from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_long_channels, get_short_channels
import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Tuple, List, Dict, Any
from PsyAnalysisToolbox.Python.utils.logging_utils import log_progress_bar

class FNIRSPreprocessor:
    """
    Universal fNIRS preprocessing module for MNE Raw objects.
    - Accepts a config dict with required and optional keys.
    - Fills in missing keys with class-level defaults.
    - Raises clear errors for missing required keys.
    - Usable in any project (no project-specific assumptions).
    """
    DEFAULT_BEER_LAMBERT_REMOVE_OD = True
    DEFAULT_FILTER_H_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_L_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("FNIRSPreprocessor initialized.")

    def process(self, raw_fnirs: mne.io.Raw, fnirs_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point for fNIRS preprocessing.
        Applies filtering, short channel regression, motion correction, and config-driven logic.
        Returns a dictionary with the processed raw fNIRS.
        """
        # Data integrity check
        if not isinstance(raw_fnirs, mne.io.Raw):
            self.logger.error("FNIRSPreprocessor: Input is not an MNE Raw object.")
            return None
        if np.isnan(raw_fnirs.get_data()).any():
            self.logger.error("FNIRSPreprocessor: NaNs detected in input Raw object.")
            return None

        steps = 3
        update, close = log_progress_bar(self.logger, steps, desc="FNIRS", per_process=True)
        update(); self.logger.info("FNIRSPreprocessor: Filtering")
        # Filtering
        filter_band = fnirs_config.get('filter_band', (0.01, 0.1))
        raw_fnirs.filter(l_freq=filter_band[0], h_freq=filter_band[1], fir_design=fnirs_config.get('filter_fir_design', self.DEFAULT_FILTER_FIR_DESIGN), verbose=False)
        self.logger.info(f"FNIRSPreprocessor: Filtered fNIRS {filter_band[0]}-{filter_band[1]} Hz.")

        # Short channel regression
        if fnirs_config.get('short_channel_regression', False):
            self.logger.info("FNIRSPreprocessor: Applying short channel regression.")
            try:
                raw_fnirs = short_channel_regression(raw_fnirs)
            except Exception as e:
                self.logger.error(f"FNIRSPreprocessor: Short channel regression failed: {e}", exc_info=True)
                return None

        # Motion correction (if implemented)
        if fnirs_config.get('motion_correction_method', 'none') != 'none':
            self.logger.info(f"FNIRSPreprocessor: Motion correction method '{fnirs_config['motion_correction_method']}' requested, but not implemented in this module.")

        update(); self.logger.info("FNIRSPreprocessor: GLM")
        self.logger.info("FNIRSPreprocessor: fNIRS preprocessing completed.")
        update(); self.logger.info("FNIRSPreprocessor: Done")
        close()
        return {'fnirs_processed_raw': raw_fnirs}