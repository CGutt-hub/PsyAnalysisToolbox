import os
import pyxdf
import mne
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

class XDFReader:
    # Class-level defaults for XDFReader stream names
    DEFAULT_EEG_STREAM_NAME = "DefaultEEGStreamName"
    DEFAULT_FNIRS_STREAM_NAME = "DefaultFNIRSStreamName"
    DEFAULT_ECG_STREAM_NAME = "DefaultECGStreamName"
    DEFAULT_EDA_STREAM_NAME = "DefaultEDAStreamName"
    DEFAULT_MARKER_STREAM_NAME = "DefaultMarkerStreamName"

    def __init__(self, logger,
                 eeg_stream_name: Optional[str] = None,
                 fnirs_stream_name: Optional[str] = None,
                 ecg_stream_name: Optional[str] = None,
                 eda_stream_name: Optional[str] = None,
                 marker_stream_name: Optional[str] = None
                ):
        """
        Initializes the XDFReader with specific or default stream names.

        Args:
            logger (logging.Logger): Logger instance.
            eeg_stream_name (Optional[str]): Name of the EEG stream.
            fnirs_stream_name (Optional[str]): Name of the fNIRS stream.
            ecg_stream_name (Optional[str]): Name of the ECG stream.
            eda_stream_name (Optional[str]): Name of the EDA stream.
            marker_stream_name (Optional[str]): Name of the Marker stream.
        """
        self.logger = logger

        # Helper to reduce repetition in __init__
        def _configure_stream_name(user_provided_name: Optional[str], default_name: str, name_for_log: str) -> str:
            if user_provided_name is None:
                return default_name
            if isinstance(user_provided_name, str) and user_provided_name.strip():
                return user_provided_name
            
            self.logger.warning(
                f"XDFReader: Invalid value ('{user_provided_name}') for '{name_for_log}'. "
                f"Using default: '{default_name}'."
            )
            return default_name

        self._eeg_stream_name = _configure_stream_name(eeg_stream_name, self.DEFAULT_EEG_STREAM_NAME, 'eeg_stream_name')
        self._fnirs_stream_name = _configure_stream_name(fnirs_stream_name, self.DEFAULT_FNIRS_STREAM_NAME, 'fnirs_stream_name')
        self._ecg_stream_name = _configure_stream_name(ecg_stream_name, self.DEFAULT_ECG_STREAM_NAME, 'ecg_stream_name')
        self._eda_stream_name = _configure_stream_name(eda_stream_name, self.DEFAULT_EDA_STREAM_NAME, 'eda_stream_name')
        self._marker_stream_name = _configure_stream_name(marker_stream_name, self.DEFAULT_MARKER_STREAM_NAME, 'marker_stream_name')
        
        self.logger.info(
            f"XDFReader initialized. Stream names: "
            f"EEG='{self._eeg_stream_name}', fNIRS='{self._fnirs_stream_name}', "
            f"ECG='{self._ecg_stream_name}', EDA='{self._eda_stream_name}', "
            f"Marker='{self._marker_stream_name}'."
        )

    def load_participant_streams(self, participant_id: str, xdf_file_path: str) -> Dict[str, Any]:
        """
        Loads and processes streams from an XDF file into a dictionary of DataFrames.

        Args:
            participant_id (str): The ID of the participant.
            xdf_file_path (str): The full path to the XDF file.

        Returns:
            Dict[str, Any]: A dictionary containing DataFrames for each successfully loaded and processed stream.
        """
        self.logger.info(f"XDFReader - Loading data for P:{participant_id} from {xdf_file_path}")
        
        if not os.path.exists(xdf_file_path):
            self.logger.error(f"XDFReader - XDF file not found: {xdf_file_path}")
            return {}

        try:
            streams, header = pyxdf.load_xdf(xdf_file_path)
            self.logger.info(f"XDFReader - Successfully parsed XDF file. Found {len(streams)} streams.")
        except Exception as e:
            self.logger.error(f"XDFReader - Critical error parsing XDF file {xdf_file_path}: {e}", exc_info=True)
            return {}

        stream_map = {stream['info']['name'][0]: stream for stream in streams}
        loaded_data: Dict[str, Any] = {}

        # --- Process EEG Stream ---
        if self._eeg_stream_name in stream_map:
            eeg_df = self._process_eeg_stream(stream_map[self._eeg_stream_name])
            if eeg_df is not None:
                loaded_data['eeg_df'] = eeg_df

        # --- Process ECG Stream ---
        if self._ecg_stream_name in stream_map:
            ecg_df = self._process_timeseries_stream(stream_map[self._ecg_stream_name], 'ecg_signal')
            if ecg_df is not None:
                loaded_data['ecg_df'] = ecg_df

        # --- Process EDA Stream ---
        if self._eda_stream_name in stream_map:
            eda_df = self._process_timeseries_stream(stream_map[self._eda_stream_name], 'eda_signal')
            if eda_df is not None:
                loaded_data['eda_df'] = eda_df

        # --- Process fNIRS Stream ---
        if self._fnirs_stream_name in stream_map:
            fnirs_df = self._process_fnirs_stream(stream_map[self._fnirs_stream_name])
            if fnirs_df is not None:
                loaded_data['fnirs_od_df'] = fnirs_df
                # Also store the start time for alignment, which is critical for GLM
                loaded_data['fnirs_stream_start_time_xdf'] = stream_map[self._fnirs_stream_name]['time_stamps'][0]
        
        # --- Process Marker Stream ---
        if self._marker_stream_name in stream_map:
            marker_df = self._process_marker_stream(stream_map[self._marker_stream_name])
            if marker_df is not None:
                loaded_data['xdf_markers_df'] = marker_df

        self.logger.info(f"XDFReader - Finished loading for P:{participant_id}. Loaded data keys: {list(loaded_data.keys())}")
        return loaded_data

    def _process_eeg_stream(self, stream: Dict) -> Optional[pd.DataFrame]:
        self.logger.info(f"XDFReader - Processing EEG stream: {stream['info']['name'][0]}")
        try:
            data = np.array(stream['time_series']).T
            sfreq = float(stream['info']['nominal_srate'][0])
            if sfreq <= 0: raise ValueError("Sampling rate must be positive.")
            
            ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
            ch_types = 'eeg'
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            raw_eeg = mne.io.RawArray(data, info, verbose=False)
            
            eeg_df = raw_eeg.to_data_frame()
            eeg_df.rename(columns={'time': 'time_sec'}, inplace=True)
            
            if len(eeg_df) == len(stream['time_stamps']):
                eeg_df['time_xdf'] = stream['time_stamps']
            else:
                self.logger.warning("EEG time_stamps length mismatch. 'time_xdf' column will be NaN.")
                eeg_df['time_xdf'] = np.nan

            self.logger.info(f"XDFReader - EEG stream processed into DataFrame shape {eeg_df.shape}.")
            return eeg_df
        except Exception as e:
            self.logger.warning(f"XDFReader - Failed to process EEG stream '{stream['info']['name'][0]}': {e}", exc_info=True)
            return None

    def _process_timeseries_stream(self, stream: Dict, signal_name: str) -> Optional[pd.DataFrame]:
        self.logger.info(f"XDFReader - Processing time series stream: {stream['info']['name'][0]} as '{signal_name}'")
        try:
            signal = np.array(stream['time_series']).flatten()
            sfreq = float(stream['info']['nominal_srate'][0])
            if sfreq <= 0: raise ValueError("Sampling rate must be positive.")
            
            df = pd.DataFrame({signal_name: signal})
            df['time_xdf'] = stream['time_stamps']
            df['time_sec'] = np.arange(len(df)) / sfreq
            
            self.logger.info(f"XDFReader - {signal_name} stream processed into DataFrame shape {df.shape}.")
            return df
        except Exception as e:
            self.logger.warning(f"XDFReader - Failed to process stream '{stream['info']['name'][0]}': {e}", exc_info=True)
            return None

    def _process_fnirs_stream(self, stream: Dict) -> Optional[pd.DataFrame]:
        self.logger.info(f"XDFReader - Processing fNIRS stream: {stream['info']['name'][0]}")
        try:
            data = np.array(stream['time_series']).T
            sfreq = float(stream['info']['nominal_srate'][0])
            if sfreq <= 0: raise ValueError("Sampling rate must be positive.")
            ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
            ch_types = 'fnirs_od'
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            raw_fnirs = mne.io.RawArray(data, info, verbose=False)

            fnirs_df = raw_fnirs.to_data_frame()
            fnirs_df.rename(columns={'time': 'time_sec'}, inplace=True)
            if len(fnirs_df) == len(stream['time_stamps']):
                fnirs_df['time_xdf'] = stream['time_stamps']
            else:
                self.logger.warning("fNIRS time_stamps length mismatch. 'time_xdf' column will be NaN.")
                fnirs_df['time_xdf'] = np.nan
            
            self.logger.info(f"XDFReader - fNIRS stream processed into DataFrame shape {fnirs_df.shape}.")
            return fnirs_df
        except Exception as e:
            self.logger.warning(f"XDFReader - Failed to process fNIRS stream '{stream['info']['name'][0]}': {e}", exc_info=True)
            return None

    def _process_marker_stream(self, stream: Dict) -> Optional[pd.DataFrame]:
        self.logger.info(f"XDFReader - Processing Marker stream: {stream['info']['name'][0]}")
        try:
            marker_timestamps = stream['time_stamps']
            # pyxdf wraps markers in a list, e.g., [['S 10']]
            marker_values_raw = [val[0] if val and isinstance(val, list) else val for val in stream['time_series']]
            marker_df = pd.DataFrame({'timestamp': marker_timestamps, 'marker_value': marker_values_raw})
            self.logger.info(f"XDFReader - Marker stream processed into DataFrame shape {marker_df.shape}.")
            return marker_df
        except Exception as e:
            self.logger.warning(f"XDFReader - Failed to process Marker stream '{stream['info']['name'][0]}': {e}", exc_info=True)
            return None
