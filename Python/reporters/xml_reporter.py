import pandas as pd
from typing import Optional, Dict, Any
import os
import logging

class XMLReporter:
    """
    Reporter for saving DataFrames to XML files.
    Input: DataFrame (data_df)
    Output: XML file (side effect)
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("XMLReporter initialized.")

    def save_dataframe(self, data_df, output_dir, filename, **kwargs):
        if not isinstance(data_df, pd.DataFrame):
            self.logger.error('XMLReporter: data_df must be a DataFrame.')
            return None
        if not filename.lower().endswith('.xml'):
            self.logger.warning(f"Filename '{filename}' does not end with .xml. Appending .xml extension.")
            filename += '.xml'

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        try:
            # Sensible defaults for to_xml
            xml_params = {'index': False, 'root_name': 'data', 'row_name': 'item', 'parser': 'lxml'}
            xml_params.update(kwargs) # Allow user to override defaults

            data_df.to_xml(file_path, **xml_params)
            self.logger.info(f"Successfully saved DataFrame to XML: {file_path}")
            return file_path
        except ImportError:
            self.logger.error("The 'lxml' library is required to write XML files. Please install it using 'pip install lxml'.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to XML at {file_path}: {e}", exc_info=True)
            return None