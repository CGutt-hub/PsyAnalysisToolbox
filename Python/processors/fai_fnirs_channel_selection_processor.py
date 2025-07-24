import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging

class FAIFNIRSChannelSelectionProcessor:
    """
    Selects EEG channels for FAI analysis based on significant fNIRS GLM results.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("FAIFNIRSChannelSelector initialized.")

    def select_channels(self,
                        glm_results_df: Optional[pd.DataFrame],
                        config: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Selects a pair of EEG channels for FAI analysis based on the most significant fNIRS GLM results.

        Args:
            glm_results_df: DataFrame with fNIRS GLM results.
            config: Dictionary with configuration parameters.

        Returns:
            A tuple (left_eeg_channel, right_eeg_channel), or None if selection fails.
        """
        self.logger.info("Attempting to select FAI EEG channels based on fNIRS GLM results.")
        if glm_results_df is None or glm_results_df.empty:
            self.logger.info("fNIRS GLM results not available, cannot perform dynamic FAI channel selection.")
            return None

        try:
            contrast, p_thresh = config['contrast'], config['p_thresh']
            fnirs_eeg_map, left_hemi, right_hemi = config['fnirs_eeg_map'], config['left_hemi_fnirs'], config['right_hemi_fnirs']

            sig_results = glm_results_df[(glm_results_df['Contrast'] == contrast) & (glm_results_df['p-value'] < p_thresh)].copy()
            if sig_results.empty:
                self.logger.warning(f"No significant fNIRS channels for contrast '{contrast}' at p < {p_thresh}. Cannot guide FAI.")
                return None

            sig_results['sd_pair'] = sig_results['Channel'].apply(lambda x: x.split(' ')[0].replace('_', '-'))
            best_left_series = sig_results[sig_results['sd_pair'].isin(left_hemi)].nsmallest(1, 'p-value')
            best_right_series = sig_results[sig_results['sd_pair'].isin(right_hemi)].nsmallest(1, 'p-value')

            if best_left_series.empty or best_right_series.empty:
                self.logger.warning("Could not find a significant fNIRS channel in both left and right hemispheres. Cannot guide FAI.")
                return None

            left_eeg, right_eeg = fnirs_eeg_map.get(best_left_series.iloc[0]['sd_pair']), fnirs_eeg_map.get(best_right_series.iloc[0]['sd_pair'])
            if left_eeg and right_eeg:
                final_left, final_right = left_eeg.split(',')[0].strip(), right_eeg.split(',')[0].strip()
                self.logger.info(f"fNIRS-guided selection successful. Using EEG pair: ({final_left}, {final_right}) for FAI.")
                return (final_left, final_right)
            self.logger.warning("Found significant fNIRS channels, but could not map them to EEG channels using [FNIRS_EEG_MAP].")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error during fNIRS-guided channel selection: Missing config or data issue. {e}", exc_info=True)
            return None