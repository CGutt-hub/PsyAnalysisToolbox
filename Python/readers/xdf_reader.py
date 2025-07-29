"""
XDF Reader Module
----------------
Reads and parses XDF (Extensible Data Format) files for multimodal data.
Config-driven, robust, and maintainable.
"""
import pyxdf
import pandas as pd
import logging
from typing import Dict, Any, Optional

class XDFReader:
    """
    Reads and parses XDF (Extensible Data Format) files for multimodal data.
    - Accepts config dict for reading parameters.
    - Returns parsed data as a dictionary of DataFrames or arrays.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger, eeg_stream_name: Optional[str] = None, fnirs_stream_name: Optional[str] = None, marker_stream_name: Optional[str] = None):
        self.logger = logger
        self.eeg_stream_name = eeg_stream_name
        self.fnirs_stream_name = fnirs_stream_name
        self.marker_stream_name = marker_stream_name
        self.logger.info("XDFReader initialized.")

    def load_participant_streams(self, participant_id: str, xdf_path: str) -> Dict[str, Any]:
        """
        Loads and parses an XDF file for the given participant.
        Returns a dictionary with stream names as keys and parsed data as values.
        """
        self.logger.info(f"XDFReader: Loading streams for participant {participant_id} from {xdf_path}")
        try:
            streams, header = pyxdf.load_xdf(xdf_path)
        except Exception as e:
            self.logger.error(f"XDFReader: Failed to load XDF file: {e}", exc_info=True)
            return {}
        stream_dict = {}
        for stream in streams:
            name = stream['info']['name'][0]
            time_series = stream['time_series']
            # Try to convert to DataFrame if possible
            try:
                df = pd.DataFrame(time_series)
                df.columns = [f"ch_{i}" for i in range(df.shape[1])] if df.shape[1] > 1 else [name]
                stream_dict[name] = df
                self.logger.info(f"XDFReader: Loaded stream '{name}' with shape {df.shape}.")
            except Exception as e:
                stream_dict[name] = time_series
                self.logger.warning(f"XDFReader: Loaded stream '{name}' as array (could not convert to DataFrame): {e}")
        return stream_dict
