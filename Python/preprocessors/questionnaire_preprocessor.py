"""
Questionnaire Preprocessor Module
--------------------------------
Universal questionnaire preprocessing for tabular data.
Handles data cleaning, scoring, and config-driven logic.
Config-driven, robust, and maintainable.
"""
import pandas as pd
import logging
from typing import Dict, Any, Optional

class QuestionnairePreprocessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("QuestionnairePreprocessor initialized.")

    def process(self, questionnaire_df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(questionnaire_df, pd.DataFrame):
            self.logger.error("QuestionnairePreprocessor: Input is not a pandas DataFrame.")
            return None
        self.logger.info(f"QuestionnairePreprocessor: Initial data shape: {questionnaire_df.shape}")
        cleaned_df = questionnaire_df.dropna()
        self.logger.info(f"QuestionnairePreprocessor: After dropna, data shape: {cleaned_df.shape}")
        if 'score_columns' in config:
            score_cols = config['score_columns']
            cleaned_df['mean_score'] = cleaned_df[score_cols].mean(axis=1)
            self.logger.info(f"QuestionnairePreprocessor: Computed mean score for columns: {score_cols}")
        return {'questionnaire_processed_df': cleaned_df}