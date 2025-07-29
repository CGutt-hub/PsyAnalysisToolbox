"""
Score Analyzer Module
--------------------
Computes questionnaire or behavioral scores from input data.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class ScoreAnalyzer:
    """
    Computes questionnaire or behavioral scores from input data.
    - Accepts config dict for scoring parameters.
    - Returns scores as a DataFrame.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("ScoreAnalyzer initialized.")

    def compute_scores(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Computes scores from the provided data using config parameters.
        Returns a DataFrame with computed scores.
        """
        if data is None or data.empty:
            self.logger.error("ScoreAnalyzer: No data provided.")
            return pd.DataFrame([], columns=pd.Index(['score_name', 'value']))
        try:
            method = config.get('scoring_method', 'mean')
            results = []
            for col in data.columns:
                if method == 'sum':
                    val = data[col].sum()
                else:
                    val = data[col].mean()
                results.append({'score_name': col, 'value': val})
            self.logger.info(f"ScoreAnalyzer: Computed scores for {len(results)} columns.")
            return pd.DataFrame(results)
        except Exception as e:
            self.logger.error(f"ScoreAnalyzer: Failed to compute scores: {e}")
            return pd.DataFrame([], columns=pd.Index(['score_name', 'value']))