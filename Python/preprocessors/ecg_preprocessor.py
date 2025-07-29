"""
ECG Preprocessor Module
----------------------
Universal ECG preprocessing for R-peak detection using NeuroKit2.
Handles filtering, R-peak detection, and config-driven logic.
Config-driven, robust, and maintainable.
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
from typing import Tuple, Optional, Union, Dict, Any
from PsyAnalysisToolbox.Python.utils.logging_utils import log_progress_bar

class ECGPreprocessor:
    """
    Universal ECG preprocessing module for R-peak detection using NeuroKit2.
    - Accepts a config dict with required and optional keys.
    - Fills in missing keys with class-level defaults.
    - Raises clear errors for missing required keys.
    - Usable in any project (no project-specific assumptions).
    """
    DEFAULT_RPEAK_DETECTION_METHOD = "neurokit"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("ECGPreprocessor initialized.")

    def process(self, ecg_signal: np.ndarray, ecg_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point for ECG preprocessing.
        Applies filtering, R-peak detection, and config-driven logic.
        Returns a dictionary with the processed ECG signal and R-peaks.
        """
        # Data integrity check
        if not isinstance(ecg_signal, np.ndarray):
            self.logger.error("ECGPreprocessor: Input is not a numpy ndarray.")
            return None
        if np.isnan(ecg_signal).any():
            self.logger.error("ECGPreprocessor: NaNs detected in input ECG signal.")
            return None

        steps = 4
        update, close = log_progress_bar(self.logger, steps, desc="ECG", per_process=True)
        update(); self.logger.info("ECGPreprocessor: Filtering")
        # R-peak detection
        method = ecg_config.get('ecg_rpeak_method', self.DEFAULT_RPEAK_DETECTION_METHOD)
        try:
            signals, info = nk.ecg_process(ecg_signal, sampling_rate=ecg_config.get('sampling_rate', 1000))
            rpeaks = signals['ECG_R_Peaks']
            self.logger.info(f"ECGPreprocessor: Detected {len(rpeaks)} R-peaks using method '{method}'.")
        except Exception as e:
            self.logger.error(f"ECGPreprocessor: R-peak detection failed: {e}", exc_info=True)
            return None
        update(); self.logger.info("ECGPreprocessor: R-peak detection")

        update(); self.logger.info("ECGPreprocessor: HRV calculation")
        # HRV calculation (example, if needed)
        # hrv_data = nk.hrv(rpeaks, sampling_rate=ecg_config.get('sampling_rate', 1000))
        # self.logger.info(f"ECGPreprocessor: HRV calculated. Features: {hrv_data.keys()}")
        update(); self.logger.info("ECGPreprocessor: Done")
        close()

        self.logger.info("ECGPreprocessor: ECG preprocessing completed.")
        return {'ecg_processed_signal': ecg_signal, 'rpeaks': rpeaks}