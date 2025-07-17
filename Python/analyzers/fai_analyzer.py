import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class FAIAnalyzer:
    # Default parameters for FAI calculation
    DEFAULT_MIN_POWER_THRESHOLD = 1e-12

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FAIAnalyzer initialized.")

    def compute_fai_from_psd_df(self,
                                psd_df: pd.DataFrame,
                                fai_band_name: str,
                                fai_electrode_pairs_config: List[Tuple[str, str]],
                                min_power_threshold: float = DEFAULT_MIN_POWER_THRESHOLD
                                ) -> Optional[pd.DataFrame]:
        """
        Computes Frontal Asymmetry Index (FAI) from a long-format PSD DataFrame.

        Args:
            psd_df (pd.DataFrame): DataFrame with PSD results, requiring columns
                                   ['condition', 'band', 'channel', 'power'].
            fai_band_name (str): The specific band to use for FAI calculation.
            fai_electrode_pairs_config (List[Tuple[str, str]]): List of tuples defining electrode pairs.
            min_power_threshold (float): Minimum power value to consider.

        Returns:
            Optional[pd.DataFrame]: DataFrame with FAI results in long format
                                    ['condition', 'pair_name', 'fai_value'], or None on error.
        """
        required_cols = ['condition', 'band', 'channel', 'power']
        if not isinstance(psd_df, pd.DataFrame) or not all(col in psd_df.columns for col in required_cols):
            self.logger.error(f"FAIAnalyzer: Input psd_df must be a DataFrame with columns {required_cols}.")
            return None

        if not isinstance(fai_band_name, str) or not fai_band_name.strip():
            self.logger.error("FAIAnalyzer: fai_band_name must be a non-empty string.")
            return None

        # Filter for the specified band
        band_psd_df = psd_df[psd_df['band'] == fai_band_name].copy()
        if band_psd_df.empty:
            self.logger.warning(f"FAIAnalyzer: No PSD data found for the specified band '{fai_band_name}'.")
            return pd.DataFrame()

        # Pivot to get channels as columns for easy access
        try: 
            psd_wide_df = band_psd_df.pivot_table(index='condition', columns='channel', values='power')
        except Exception as e:
            self.logger.error(f"FAIAnalyzer: Could not pivot the PSD DataFrame. Error: {e}", exc_info=True)
            return None

        all_fai_results = []

        for ch_left, ch_right in fai_electrode_pairs_config:
            pair_name = f"{ch_right}_vs_{ch_left}"

            if ch_left not in psd_wide_df.columns or ch_right not in psd_wide_df.columns:
                self.logger.warning(f"FAIAnalyzer: Channels for pair {pair_name} not found in the PSD data. Skipping pair.")
                continue

            # Vectorized calculation for the pair across all conditions
            power_left_series = psd_wide_df[ch_left]
            power_right_series = psd_wide_df[ch_right]

            # Create a boolean mask for valid power values
            valid_power_mask = (power_left_series > min_power_threshold) & (power_right_series > min_power_threshold)

            # Calculate FAI only where power is valid
            fai_values = pd.Series(np.nan, index=psd_wide_df.index) # Initialize with NaNs
            fai_values[valid_power_mask] = np.log(power_right_series[valid_power_mask]) - np.log(power_left_series[valid_power_mask])

            # Log warnings for invalid power
            if not valid_power_mask.all():
                invalid_conditions = valid_power_mask[~valid_power_mask].index.tolist()
                self.logger.warning(f"FAIAnalyzer: Zero/tiny power for pair {pair_name} in conditions: {invalid_conditions}. FAI set to NaN.")

            # Convert the results for this pair to a long-format DataFrame and append
            pair_results_df = fai_values.reset_index()
            pair_results_df.columns = ['condition', 'fai_value']
            pair_results_df['pair_name'] = pair_name
            all_fai_results.append(pair_results_df)

        if not all_fai_results:
            self.logger.warning("FAIAnalyzer: No FAI results were calculated.")
            return pd.DataFrame()

        final_fai_df = pd.concat(all_fai_results, ignore_index=True)
        return final_fai_df[['condition', 'pair_name', 'fai_value']] # Ensure column order

    def compute_fai_from_psd(self,
                             psd_results_all_bands: Dict[str, Dict[str, Dict[str, float]]],
                             fai_band_name: str,
                             fai_electrode_pairs_config: List[Tuple[str, str]],
                             min_power_threshold: float = DEFAULT_MIN_POWER_THRESHOLD
                             ) -> Dict[str, Dict[str, float]]:
        """
        Computes Frontal Asymmetry Index (FAI) from pre-computed PSD results.

        Args:
            psd_results_all_bands (dict): PSD results in the format {'condition': {'band': {'channel': power}}}.
            fai_band_name (str): The name of the frequency band in psd_results_all_bands to use for FAI.
            fai_electrode_pairs_config (list): List of tuples defining electrode pairs for FAI (e.g., [('F3', 'F4')]).
            min_power_threshold (float): Minimum power value to consider for FAI calculation;
                                         values below this (and <=0) lead to NaN. Defaults to FAIAnalyzer.DEFAULT_MIN_POWER_THRESHOLD.

        Returns:
            dict: FAI results in the format {'condition': {'pair_name': fai_value}}.

        Note:
            This method is deprecated in favor of compute_fai_from_psd_df for better pipeline integration.
        """
        if not isinstance(psd_results_all_bands, dict) or not psd_results_all_bands:
            self.logger.warning("FAIAnalyzer - Input psd_results_all_bands is not a non-empty dictionary. Skipping FAI computation.")
            return {}
        if not isinstance(fai_band_name, str) or not fai_band_name.strip():
            self.logger.error("FAIAnalyzer - fai_band_name must be a non-empty string. Skipping FAI computation.")
            return {}
        if not isinstance(fai_electrode_pairs_config, list) or not fai_electrode_pairs_config:
            self.logger.error("FAIAnalyzer - fai_electrode_pairs_config must be a non-empty list of electrode pairs. Skipping FAI computation.")
            return {}
        if not all(isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(ch, str) and ch.strip() for ch in pair) for pair in fai_electrode_pairs_config):
             self.logger.error("FAIAnalyzer - fai_electrode_pairs_config must be a list of tuples, where each tuple contains two non-empty strings. Skipping FAI computation.")
             return {}
        if not isinstance(min_power_threshold, (int, float)) or min_power_threshold < 0:
            self.logger.error(f"FAIAnalyzer - min_power_threshold must be a non-negative number. Got {min_power_threshold}. Using default {self.DEFAULT_MIN_POWER_THRESHOLD}.")
            min_power_threshold = self.DEFAULT_MIN_POWER_THRESHOLD

        fai_results: Dict[str, Dict[str, float]] = {}
        self.logger.debug(f"FAIAnalyzer - Computing FAI using PSD from band: {fai_band_name}")

        for condition_name, bands_data in psd_results_all_bands.items():
            fai_results[condition_name] = {}
            power_for_fai_band = bands_data.get(fai_band_name)

            if not power_for_fai_band:
                self.logger.warning(f"FAIAnalyzer - Power for FAI band '{fai_band_name}' not found in PSD results for condition '{condition_name}'. Skipping FAI for this condition.")
                continue

            for ch_left, ch_right in fai_electrode_pairs_config:
                pair_name = f"{ch_right}_vs_{ch_left}" # e.g., F4_vs_F3
                power_left = power_for_fai_band.get(ch_left)
                power_right = power_for_fai_band.get(ch_right)

                if power_left is not None and power_right is not None and isinstance(power_left, (int, float)) and isinstance(power_right, (int, float)):
                    if power_left > min_power_threshold and power_right > min_power_threshold: # Avoid log(0) or log(negative)
                        fai_val = np.log(power_right) - np.log(power_left)
                        fai_results[condition_name][pair_name] = fai_val
                        self.logger.debug(f"FAIAnalyzer - FAI {pair_name} for {condition_name} (L:{power_left:.4e}, R:{power_right:.4e}): {fai_val:.4f}")
                    else: # Either power_left OR power_right <= min_power_threshold
                        self.logger.warning(f"FAIAnalyzer - Zero/tiny power for FAI {pair_name}, {condition_name}. FAI set to NaN.")
                        fai_results[condition_name][pair_name] = np.nan
                else:
                    self.logger.warning(f"FAIAnalyzer - Missing power for FAI {pair_name} (L:{ch_left}, R:{ch_right}), {condition_name}. FAI set to NaN.")
                    fai_results[condition_name][pair_name] = np.nan
        return fai_results