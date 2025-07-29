"""
ANOVA Analyzer Module
--------------------
Performs ANOVA statistical analysis on input data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional
import pingouin as pg

class ANOVAAnalyzer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("ANOVAAnalyzer initialized.")

    def run_anova(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Runs ANOVA on the provided data using config parameters.
        Supports one-way and repeated-measures ANOVA.
        Returns a DataFrame with ANOVA results.
        """
        try:
            anova_type = config.get('anova_type', 'oneway')
            if anova_type == 'oneway':
                dv = config['dv']
                between = config['between']
                self.logger.info(f"ANOVAAnalyzer: Running one-way ANOVA on DV='{dv}' with between='{between}'.")
                results = pg.anova(data=data, dv=dv, between=between, detailed=True)
            elif anova_type == 'rm':
                dv = config['dv']
                within = config['within']
                subject = config['subject']
                self.logger.info(f"ANOVAAnalyzer: Running repeated-measures ANOVA on DV='{dv}' with within='{within}', subject='{subject}'.")
                results = pg.rm_anova(data=data, dv=dv, within=within, subject=subject, detailed=True)
            else:
                raise ValueError(f"Unsupported ANOVA type: {anova_type}")
            self.logger.info("ANOVAAnalyzer: ANOVA completed.")
            return results
        except Exception as e:
            self.logger.error(f"ANOVAAnalyzer: ANOVA failed: {e}", exc_info=True)
            raise