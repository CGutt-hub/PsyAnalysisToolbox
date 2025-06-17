import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from typing import List, Tuple, Union, Optional
import logging # Added for consistency if other classes use it

class FDRCorrector:
    """
    A class to apply statistical corrections.
    """
    DEFAULT_ALPHA = 0.05

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the StatsCorrector.

        Args:
            logger (Optional[logging.Logger]): Logger instance. If None, a default logger is used.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("StatsCorrector initialized.")

    def apply_fdr_correction(self, p_values: Union[List[float], np.ndarray], alpha: float = DEFAULT_ALPHA) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies FDR (Benjamini-Hochberg) correction to a list of p-values.
        Args:
            p_values (Union[List[float], np.ndarray]): List or array of p-values.
            alpha (float): Significance level. Defaults to StatsCorrector.DEFAULT_ALPHA.
        Returns:
            tuple: (rejected_hypotheses, corrected_p_values)
                   rejected_hypotheses is a boolean array.
                   corrected_p_values is an array of FDR-corrected p-values.
        """
        if not isinstance(p_values, (list, np.ndarray)) or len(p_values) == 0:
            self.logger.warning("apply_fdr_correction received empty p-values list/array.")
            return np.array([]), np.array([])
        if not (isinstance(alpha, float) and 0 < alpha < 1):
            self.logger.error(f"Invalid alpha value: {alpha}. Must be a float between 0 and 1 (exclusive).")
            raise ValueError("alpha must be a float between 0 and 1 (exclusive).")
        
        p_values_array = np.asarray(p_values)
        if np.all(np.isnan(p_values_array)): # Handle case where all p-values are NaN
            self.logger.info("All input p-values are NaN. Returning as is for FDR correction.")
            return np.array([False] * len(p_values_array)), p_values_array
            
        self.logger.info(f"Applying FDR correction with alpha={alpha}.")
        return fdrcorrection(p_values_array, alpha=alpha, method='indep', is_sorted=False)