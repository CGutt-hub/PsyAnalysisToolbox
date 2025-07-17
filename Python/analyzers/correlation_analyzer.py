import pandas as pd
import pingouin as pg
import numpy as np 
from typing import Union, Dict, Optional, List # For type hinting

class CorrelationAnalyzer:
    # Default parameters for correlation calculation
    DEFAULT_CORR_METHOD = 'pearson'
    DEFAULT_SERIES1_NAME = 'Series1'
    DEFAULT_SERIES2_NAME = 'Series2'
    VALID_CORR_METHODS = ['pearson', 'spearman', 'kendall', 'bicor', 'percbend', 'shepherd', 'skipped']

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("CorrelationAnalyzer initialized.")

    def calculate_correlation_from_dataframe(self, 
                                             data_df: pd.DataFrame,
                                             col1: str,
                                             col2: str,
                                             method: str = DEFAULT_CORR_METHOD) -> Optional[pd.DataFrame]:
        """
        Calculates the correlation between two columns in a Pandas DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame containing the data.
            col1 (str): Name of the first column.
            col2 (str): Name of the second column.
            method (str): Correlation method to use.

        Returns:
            Optional[pd.DataFrame]: DataFrame with correlation results, or None on error.
        """
        if not isinstance(data_df, pd.DataFrame):
            self.logger.error("CorrelationAnalyzer - Input must be a pandas DataFrame.")
            return None

        if col1 not in data_df.columns or col2 not in data_df.columns:
            self.logger.error(f"CorrelationAnalyzer - Specified columns not found in DataFrame: '{col1}' or '{col2}'.")
            return None

        final_method = self.DEFAULT_CORR_METHOD
        if method.lower() not in self.VALID_CORR_METHODS:
            self.logger.warning(f"CorrelationAnalyzer - Invalid correlation method '{method}'. Using default '{self.DEFAULT_CORR_METHOD}'.")
        else:
            final_method = method.lower()

        # Extract the series from the DataFrame
        series1 = data_df[col1]
        series2 = data_df[col2]

        # Convert numpy arrays to pandas Series for pingouin compatibility and NaN handling
        if isinstance(series1, np.ndarray): series1 = pd.Series(series1, name=col1)
        if isinstance(series2, np.ndarray): series2 = pd.Series(series2, name=col2)

        # Use the existing calculate_correlation method to do the actual calculation
        correlation_results = self.calculate_correlation(series1, series2, final_method, col1, col2)

        if correlation_results:  # if the inner method returns a dictionary, convert to DataFrame
            return pd.DataFrame([correlation_results])
        else:
            return None

    def calculate_correlation(self,
                              series1: Union[pd.Series, np.ndarray],
                              series2: Union[pd.Series, np.ndarray],
                              method: str = DEFAULT_CORR_METHOD,
                              name1: str = DEFAULT_SERIES1_NAME,
                              name2: str = DEFAULT_SERIES2_NAME) -> Optional[Dict[str, Union[float, List[float], int, str]]]:
        if not isinstance(series1, (pd.Series, np.ndarray)) or not isinstance(series2, (pd.Series, np.ndarray)):
            self.logger.error("CorrelationAnalyzer - Input must be pandas Series or numpy arrays.")
            return None

        # Validate and set default names if necessary
        if not isinstance(name1, str) or not name1.strip():
            self.logger.warning(f"CorrelationAnalyzer - Invalid name1 ('{name1}'). Using default '{self.DEFAULT_SERIES1_NAME}'.")
            name1 = self.DEFAULT_SERIES1_NAME
        if not isinstance(name2, str) or not name2.strip():
            self.logger.warning(f"CorrelationAnalyzer - Invalid name2 ('{name2}'). Using default '{self.DEFAULT_SERIES2_NAME}'.")
            name2 = self.DEFAULT_SERIES2_NAME

        final_method = self.DEFAULT_CORR_METHOD
        if method.lower() not in self.VALID_CORR_METHODS:
            self.logger.warning(f"CorrelationAnalyzer - Invalid correlation method '{method}'. Using default '{self.DEFAULT_CORR_METHOD}'.")
        else:
            final_method = method.lower()

        # Convert numpy arrays to pandas Series for pingouin compatibility and NaN handling
        if isinstance(series1, np.ndarray): series1 = pd.Series(series1, name=name1)
        if isinstance(series2, np.ndarray): series2 = pd.Series(series2, name=name2)

        # Drop NaN pairs for correlation
        combined = pd.concat([series1, series2], axis=1).dropna()
        if len(combined) < 3: # Need at least 3 pairs for meaningful correlation
            self.logger.warning(f"CorrelationAnalyzer - Insufficient valid data points after NaN removal for {name1} vs {name2} (n={len(combined)}). Returning NaN results.")
            return {'r': np.nan, 'p-val': np.nan, 'CI95%': [np.nan, np.nan], 'n': len(combined), 'var1_name': name1, 'var2_name': name2, 'method': final_method}

        try:
            self.logger.info(f"CorrelationAnalyzer - Calculating {final_method} correlation between {name1} and {name2} (n={len(combined)}).")
            corr_result = pg.corr(combined.iloc[:, 0], combined.iloc[:, 1], method=final_method)

            result_dict = corr_result.iloc[0].to_dict() # Convert first row of DataFrame to dict
            # Add original names for clarity if used independently
            result_dict['var1_name'] = name1
            result_dict['var2_name'] = name2
            #Add descriptive stats.
            result_dict['var1_mean'] = series1.mean()
            result_dict['var2_mean'] = series2.mean()
            self.logger.info(f"CorrelationAnalyzer - Result ({name1} vs {name2}): r={result_dict.get('r', np.nan):.3f}, p={result_dict.get('p-val', np.nan):.3f}")
            return result_dict
        except Exception as e:
            self.logger.error(f"CorrelationAnalyzer - Error calculating correlation between {name1} and {name2}: {e}", exc_info=True)
            return None