"""
ERP Analyzer Module
------------------
Computes Event-Related Potentials (ERP) from EEG data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class ERPAnalyzer:
    """
    Computes Event-Related Potentials (ERP) from EEG data.
    - Accepts config dict for analysis parameters.
    - Returns ERP results as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("ERPAnalyzer initialized.")

    def compute_erp(self, eeg_data: Any, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Computes ERP from the provided EEG data using config parameters.
        Returns a DataFrame with ERP results.
        """
        # Assume eeg_data is an MNE Epochs object
        import numpy as np
        if eeg_data is None:
            self.logger.error("ERPAnalyzer: No EEG data provided.")
            return pd.DataFrame([], columns=pd.Index(['condition', 'channel', 'latency', 'amplitude']))
        try:
            erp_results = []
            for cond in eeg_data.event_id:
                evoked = eeg_data[cond].average()
                for ch_idx, ch_name in enumerate(evoked.ch_names):
                    # Find peak amplitude and latency in the ERP window
                    data = evoked.data[ch_idx]
                    peak_idx = np.argmax(np.abs(data))
                    peak_amp = data[peak_idx]
                    latency = evoked.times[peak_idx]
                    erp_results.append({
                        'condition': cond,
                        'channel': ch_name,
                        'latency': latency,
                        'amplitude': peak_amp
                    })
            df = pd.DataFrame(erp_results)
            self.logger.info(f"ERPAnalyzer: Computed ERP for {len(df)} channel-condition pairs.")
            return df
        except Exception as e:
            self.logger.error(f"ERPAnalyzer: Failed to compute ERP: {e}")
            return pd.DataFrame([], columns=pd.Index(['condition', 'channel', 'latency', 'amplitude']))