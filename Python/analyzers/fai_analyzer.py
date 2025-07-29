"""
FAI Analyzer Module
------------------
Computes Frontal Alpha Asymmetry (FAI) from PSD data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional, List

class FAIAnalyzer:
    """
    Computes Frontal Alpha Asymmetry (FAI) from PSD DataFrames.
    - Accepts config dict for band/channel parameters.
    - Returns FAI results as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("FAIAnalyzer initialized.")

    def compute_fai_from_psd_df(self, psd_df: pd.DataFrame, fai_band_name: str, fai_electrode_pairs_config: List[tuple]) -> pd.DataFrame:
        """
        Computes FAI for the given PSD DataFrame, band, and electrode pairs.
        Returns a DataFrame with FAI values for each condition and band.
        """
        self.logger.info(f"FAIAnalyzer: Computing FAI for band {fai_band_name}.")
        if psd_df is None or psd_df.empty:
            self.logger.warning("FAIAnalyzer: PSD DataFrame is empty. Returning empty DataFrame.")
            return pd.DataFrame()
        results = []
        for (left_ch, right_ch) in fai_electrode_pairs_config:
            for condition in psd_df['condition'].unique():
                left_power = psd_df[(psd_df['channel'] == left_ch) & (psd_df['band'] == fai_band_name) & (psd_df['condition'] == condition)]['power']
                right_power = psd_df[(psd_df['channel'] == right_ch) & (psd_df['band'] == fai_band_name) & (psd_df['condition'] == condition)]['power']
                if not left_power.empty and not right_power.empty:
                    fai_value = float(pd.np.log(right_power.values[0]) - pd.np.log(left_power.values[0]))
                    results.append({'condition': condition, 'band': fai_band_name, 'fai_value': fai_value})
        return pd.DataFrame(results)