import os
import pandas as pd
from typing import Optional

class CSVReporter:
    """
    A generic reporter for saving pandas DataFrames to CSV files, with an option to update existing files.
    """

    def __init__(self, logger):
        """
        Initializes the CSVReporter.
        Args:
            logger: An instance of a logger for logging messages.
        """
        self.logger = logger
        self.logger.info("CSVReporter initialized.")

    def save_dataframe(self,
                         data_df: pd.DataFrame,
                         output_dir: str,
                         filename: str,
                         update_key: Optional[str] = None) -> bool:
        """
        Saves a DataFrame to a CSV file. If an update_key is provided, it will update existing rows.

        Args:
            data_df (pd.DataFrame): The DataFrame to save.
            output_dir (str): The directory where the CSV file will be saved.
            filename (str): The name of the output CSV file (e.g., "results.csv").
            update_key (Optional[str]): The column name to use for identifying rows to replace.
                                        If a key is provided, existing rows with matching key values are removed
                                        before appending the new data. If None, the file is overwritten.

        Returns:
            bool: True if the DataFrame was saved successfully, False otherwise.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            self.logger.warning(f"CSVReporter: Provided data is not a valid, non-empty DataFrame. Skipping save for {filename}.")
            return False

        if update_key and update_key not in data_df.columns:
            self.logger.error(f"CSVReporter: update_key '{update_key}' not found in the DataFrame's columns. Cannot update CSV. Skipping.")
            return False

        try:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, filename)

            if update_key and os.path.exists(report_path):
                self.logger.info(f"CSVReporter: Updating existing report: {report_path}")
                existing_df = pd.read_csv(report_path)
                keys_to_update = data_df[update_key].unique()
                existing_df = existing_df[~existing_df[update_key].isin(keys_to_update)]
                final_df = pd.concat([existing_df, data_df], ignore_index=True)
            else:
                final_df = data_df

            final_df.to_csv(report_path, index=False)
            self.logger.info(f"CSVReporter: Successfully saved DataFrame to {report_path}")
            return True
        except Exception as e:
            self.logger.error(f"CSVReporter: Failed to save DataFrame to {os.path.join(output_dir, filename)}: {e}", exc_info=True)
            return False