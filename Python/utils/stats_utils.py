import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from typing import List, Tuple, Union

def apply_fdr_correction(p_values: Union[List[float], np.ndarray], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies FDR (Benjamini-Hochberg) correction to a list of p-values.
    Args:
        p_values (Union[List[float], np.ndarray]): List or array of p-values.
        alpha (float): Significance level.
    Returns:
        tuple: (rejected_hypotheses, corrected_p_values)
               rejected_hypotheses is a boolean array.
               corrected_p_values is an array of FDR-corrected p-values.
    """
    if not isinstance(p_values, (list, np.ndarray)) or len(p_values) == 0:
        return np.array([]), np.array([])
    p_values_array = np.asarray(p_values)
    if np.all(np.isnan(p_values_array)): # Handle case where all p-values are NaN
        return np.array([False] * len(p_values_array)), p_values_array
        
    return fdrcorrection(p_values_array, alpha=alpha, method='indep', is_sorted=False)