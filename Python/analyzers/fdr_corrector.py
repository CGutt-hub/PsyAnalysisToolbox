"""
FDR Corrector Module
-------------------
Performs False Discovery Rate (FDR) correction on p-values.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional
from statsmodels.stats.multitest import fdrcorrection

class FDRCorrector:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("FDRCorrector initialized.")

    def correct(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies FDR correction to the provided data using config parameters.
        Returns a DataFrame with corrected p-values.
        """
        p_col = config.get('p_col', 'p')
        alpha = config.get('alpha', 0.05)
        if p_col not in data.columns:
            raise ValueError(f"FDRCorrector: Column '{p_col}' not found in data.")
        rejected, p_fdr = fdrcorrection(data[p_col], alpha=alpha)
        data['p_fdr'] = p_fdr
        data['rejected'] = rejected
        self.logger.info(f"FDRCorrector: Applied FDR correction to column '{p_col}'.")
        return data