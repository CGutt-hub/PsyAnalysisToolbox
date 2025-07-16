import numpy as np
from typing import Dict, List, Tuple

class FAIAnalyzer:
    # Default parameters for FAI calculation
    DEFAULT_MIN_POWER_THRESHOLD = 1e-12

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FAIAnalyzer initialized.")

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