"""
HRV Analyzer Module
------------------
Computes Heart Rate Variability (HRV) metrics from ECG data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class HRVAnalyzer:
    """
    Computes Heart Rate Variability (HRV) metrics from ECG data.
    - Accepts config dict for analysis parameters.
    - Returns HRV results as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("HRVAnalyzer initialized.")

    def compute_hrv(self, ecg_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Computes HRV metrics from the provided ECG data using config parameters.
        Returns a DataFrame with HRV results.
        """
        import numpy as np
        if ecg_data is None or ecg_data.empty:
            self.logger.error("HRVAnalyzer: No ECG data provided.")
            return pd.DataFrame([], columns=pd.Index(['metric', 'value']))
        try:
            # Try to get R-peak sample indices
            if 'R_Peak_Sample' in ecg_data.columns:
                rpeaks = ecg_data['R_Peak_Sample'].to_numpy()
            elif 'rpeaks' in ecg_data.columns:
                rpeaks = ecg_data['rpeaks'].to_numpy()
            else:
                self.logger.error("HRVAnalyzer: No R-peak column found in ECG data.")
                return pd.DataFrame([], columns=pd.Index(['metric', 'value']))
            # Try both nested and flat config
            try:
                if 'ECG' in config and isinstance(config['ECG'], dict):
                    sfreq = float(config['ECG'].get('sfreq', 1000.0))
                else:
                    sfreq = float(config.get('ECG.sfreq', 1000.0))
            except Exception:
                sfreq = 1000.0
            rr_intervals = np.diff(rpeaks) / sfreq
            mean_rr = np.mean(rr_intervals)
            sdnn = np.std(rr_intervals)
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            results = [
                {'metric': 'mean_rr', 'value': mean_rr},
                {'metric': 'sdnn', 'value': sdnn},
                {'metric': 'rmssd', 'value': rmssd}
            ]
            self.logger.info("HRVAnalyzer: Computed HRV metrics.")
            return pd.DataFrame(results)
        except Exception as e:
            self.logger.error(f"HRVAnalyzer: Failed to compute HRV: {e}")
            return pd.DataFrame([], columns=pd.Index(['metric', 'value']))