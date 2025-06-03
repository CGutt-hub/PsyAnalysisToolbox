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

                if power_left is not None and power_right is not None:
                    if power_left > min_power_threshold and power_right > min_power_threshold: # Avoid log(0) or log(negative)
                        fai_val = np.log(power_right) - np.log(power_left)
                        fai_results[condition_name][pair_name] = fai_val
                        self.logger.debug(f"FAIAnalyzer - FAI {pair_name} for {condition_name}: {fai_val:.4f}")
                    else:
                        self.logger.warning(f"FAIAnalyzer - Zero/tiny power for FAI {pair_name}, {condition_name}. FAI set to NaN.")
                        fai_results[condition_name][pair_name] = np.nan
                else:
                    self.logger.warning(f"FAIAnalyzer - Missing power for FAI {pair_name} (L:{ch_left}, R:{ch_right}), {condition_name}. FAI set to NaN.")
                    fai_results[condition_name][pair_name] = np.nan
        return fai_results