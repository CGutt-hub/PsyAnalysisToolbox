import os
import pyxdf
import mne
import numpy as np
import pandas as pd
from typing import Optional

# Module-level defaults for XDFReader stream names
XDF_READER_DEFAULT_EEG_STREAM_NAME = "DefaultEEGStreamName"
XDF_READER_DEFAULT_FNIRS_STREAM_NAME = "DefaultFNIRSStreamName"
XDF_READER_DEFAULT_ECG_STREAM_NAME = "DefaultECGStreamName"
XDF_READER_DEFAULT_EDA_STREAM_NAME = "DefaultEDAStreamName"
XDF_READER_DEFAULT_MARKER_STREAM_NAME = "DefaultMarkerStreamName"

class XDFReader:
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
            eeg_stream_name (Optional[str]): Name of the EEG stream in XDF.
                                             Defaults to XDF_READER_DEFAULT_EEG_STREAM_NAME.
            fnirs_stream_name (Optional[str]): Name of the fNIRS stream in XDF.
                                               Defaults to XDF_READER_DEFAULT_FNIRS_STREAM_NAME.
            ecg_stream_name (Optional[str]): Name of the ECG stream in XDF.
                                             Defaults to XDF_READER_DEFAULT_ECG_STREAM_NAME.
            eda_stream_name (Optional[str]): Name of the EDA stream in XDF.
                                             Defaults to XDF_READER_DEFAULT_EDA_STREAM_NAME.
            marker_stream_name (Optional[str]): Name of the Marker stream in XDF.
                                                Defaults to XDF_READER_DEFAULT_MARKER_STREAM_NAME.
        """
        self.logger = logger

        # Configure EEG stream name
        self._eeg_stream_name = XDF_READER_DEFAULT_EEG_STREAM_NAME
        if eeg_stream_name is not None:
            if isinstance(eeg_stream_name, str) and eeg_stream_name.strip():
                self._eeg_stream_name = eeg_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{eeg_stream_name}') provided for 'eeg_stream_name'. "
                    f"Using default: '{XDF_READER_DEFAULT_EEG_STREAM_NAME}'."
                )

        # Configure fNIRS stream name
        self._fnirs_stream_name = XDF_READER_DEFAULT_FNIRS_STREAM_NAME
        if fnirs_stream_name is not None:
            if isinstance(fnirs_stream_name, str) and fnirs_stream_name.strip():
                self._fnirs_stream_name = fnirs_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{fnirs_stream_name}') provided for 'fnirs_stream_name'. "
                    f"Using default: '{XDF_READER_DEFAULT_FNIRS_STREAM_NAME}'."
                )

        # Configure ECG stream name
        self._ecg_stream_name = XDF_READER_DEFAULT_ECG_STREAM_NAME
        if ecg_stream_name is not None:
            if isinstance(ecg_stream_name, str) and ecg_stream_name.strip():
                self._ecg_stream_name = ecg_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{ecg_stream_name}') provided for 'ecg_stream_name'. "
                    f"Using default: '{XDF_READER_DEFAULT_ECG_STREAM_NAME}'."
                )

        # Configure EDA stream name
        self._eda_stream_name = XDF_READER_DEFAULT_EDA_STREAM_NAME
        if eda_stream_name is not None:
            if isinstance(eda_stream_name, str) and eda_stream_name.strip():
                self._eda_stream_name = eda_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{eda_stream_name}') provided for 'eda_stream_name'. "
                    f"Using default: '{XDF_READER_DEFAULT_EDA_STREAM_NAME}'."
                )

        # Configure Marker stream name
        self._marker_stream_name = XDF_READER_DEFAULT_MARKER_STREAM_NAME
        if marker_stream_name is not None:
            if isinstance(marker_stream_name, str) and marker_stream_name.strip():
                self._marker_stream_name = marker_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{marker_stream_name}') provided for 'marker_stream_name'. "
                    f"Using default: '{XDF_READER_DEFAULT_MARKER_STREAM_NAME}'."
                )

        self.logger.info(
            f"XDFReader initialized. Stream names configured: "
            f"EEG='{self._eeg_stream_name}', fNIRS='{self._fnirs_stream_name}', "
            f"ECG='{self._ecg_stream_name}', EDA='{self._eda_stream_name}', "
            f"Marker='{self._marker_stream_name}'."
        )

    def load_participant_streams(self, participant_id: str, xdf_file_path: str):
        """
        Loads relevant data streams (EEG, ECG, EDA, fNIRS, Markers) from XDF files
        for a given participant.

        Args:
            participant_id (str): The ID of the participant.
            xdf_file_path (str): The full path to the XDF file for the participant.

        Returns:
            dict: A dictionary containing loaded MNE Raw objects or signal arrays,
                  their sampling frequencies, and raw XDF markers.
                  Returns None for modalities/streams not found.
        """
        self.logger.info(f"XDFReader - Loading data for participant {participant_id} from {xdf_file_path}")
        loaded_data = {}

        if not os.path.exists(xdf_file_path):
            self.logger.error(f"XDFReader - XDF file not found: {xdf_file_path}")
            return loaded_data

        try:
            streams, header = pyxdf.load_xdf(xdf_file_path)
            self.logger.info(f"XDFReader - Successfully loaded XDF file. Found {len(streams)} streams.")

            stream_map = {stream['info']['name'][0]: stream for stream in streams}
            raw_eeg = None # Initialize to handle cases where EEG is not loaded first
            processed_stream_names = set() # Keep track of streams handled by specific loaders

            # --- Load EEG ---
            if self._eeg_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading EEG stream: {self._eeg_stream_name}")
                stream = stream_map[self._eeg_stream_name]
                try:
                    # Assuming EEG data is in stream['time_series'] and is float/double
                    # Assuming channel names are in stream['info']['desc'][0]['channels'][0]['channel']
                    data = np.array(stream['time_series']).T # Transpose to be channels x samples
                    sfreq = float(stream['info']['nominal_srate'][0])
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['eeg'] * len(ch_names)
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) # type: ignore
                    raw_eeg = mne.io.RawArray(data, info, verbose=False)

                    loaded_data['eeg'] = raw_eeg
                    processed_stream_names.add(self._eeg_stream_name)
                    self.logger.info(f"XDFReader - Loaded EEG data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"XDFReader - Error loading or processing EEG stream: {e}", exc_info=True)
                    loaded_data['eeg'] = None
            else:
                self.logger.warning(f"XDFReader - EEG stream '{self._eeg_stream_name}' not found.")
                loaded_data['eeg'] = None

            # --- Load ECG ---
            if self._ecg_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading ECG stream: {self._ecg_stream_name}")
                stream = stream_map[self._ecg_stream_name]
                try:
                    # Assuming ECG is a single channel time series
                    ecg_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    ecg_sfreq = float(stream['info']['nominal_srate'][0])
                    ecg_times = stream['time_stamps'] # Absolute timestamps from XDF

                    loaded_data['ecg_signal'] = ecg_signal
                    loaded_data['ecg_sfreq'] = ecg_sfreq
                    loaded_data['ecg_times'] = ecg_times # Store absolute times for alignment
                    processed_stream_names.add(self._ecg_stream_name)
                    self.logger.info(f"XDFReader - Loaded ECG data at {ecg_sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"XDFReader - Error loading or processing ECG stream: {e}", exc_info=True)
                    loaded_data['ecg_signal'] = None
                    loaded_data['ecg_sfreq'] = None
                    loaded_data['ecg_times'] = None
            else:
                self.logger.warning(f"XDFReader - ECG stream '{self._ecg_stream_name}' not found.")
                loaded_data['ecg_signal'] = None
                loaded_data['ecg_sfreq'] = None
                loaded_data['ecg_times'] = None


            # --- Load EDA ---
            if self._eda_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading EDA stream: {self._eda_stream_name}")
                stream = stream_map[self._eda_stream_name]
                try:
                    # Assuming EDA is a single channel time series
                    eda_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    eda_sfreq = float(stream['info']['nominal_srate'][0])
                    eda_times = stream['time_stamps'] # Absolute timestamps from XDF

                    loaded_data['eda_signal'] = eda_signal
                    loaded_data['eda_sfreq'] = eda_sfreq
                    loaded_data['eda_times'] = eda_times # Store absolute times for alignment
                    processed_stream_names.add(self._eda_stream_name)
                    self.logger.info(f"XDFReader - Loaded EDA data at {eda_sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"XDFReader - Error loading or processing EDA stream: {e}", exc_info=True)
                    loaded_data['eda_signal'] = None
                    loaded_data['eda_sfreq'] = None
                    loaded_data['eda_times'] = None
            else:
                self.logger.warning(f"XDFReader - EDA stream '{self._eda_stream_name}' not found.")
                loaded_data['eda_signal'] = None
                loaded_data['eda_sfreq'] = None
                loaded_data['eda_times'] = None

            # --- Load fNIRS ---
            raw_fnirs_od = None # Initialize here
            fnirs_stream_start_time_xdf = None # Initialize here

            if self._fnirs_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading fNIRS stream: {self._fnirs_stream_name}")
                stream = stream_map[self._fnirs_stream_name]
                try:
                    data = np.array(stream['time_series']).T 
                    fnirs_stream_start_time_xdf = stream['time_stamps'][0] # Absolute XDF time of the first sample
                    sfreq = float(stream['info']['nominal_srate'][0])
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['fnirs_od'] * len(ch_names)

                    # MNE expects ch_types as a list of strings, one for each channel
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) # type: ignore
                    raw_fnirs_od = mne.io.RawArray(data, info, verbose=False)
                    self.logger.info(f"XDFReader - Loaded fNIRS data with {len(ch_names)} channels at {sfreq} Hz.")
                    processed_stream_names.add(self._fnirs_stream_name)

                except Exception as e:
                    raw_fnirs_od = None # Already initialized, but good to be explicit on error path
                    fnirs_stream_start_time_xdf = None # Already initialized
                    self.logger.error(f"XDFReader - Error loading or processing fNIRS stream: {e}", exc_info=True)
            else:
                self.logger.warning(f"XDFReader - fNIRS stream '{self._fnirs_stream_name}' not found.")
                # raw_fnirs_od and fnirs_stream_start_time_xdf remain None as initialized

            loaded_data['fnirs_od'] = raw_fnirs_od
            loaded_data['fnirs_stream_start_time_xdf'] = fnirs_stream_start_time_xdf

            # --- Load Raw XDF Markers ---
            xdf_markers_df = pd.DataFrame() # Initialize in case marker stream is not found
            if self._marker_stream_name in stream_map:
                 marker_stream = stream_map[self._marker_stream_name]
                 marker_timestamps = marker_stream['time_stamps'] # Absolute XDF times
                 marker_values_raw = [val[0] for val in marker_stream['time_series']]

                 xdf_markers_df = pd.DataFrame({
                     'timestamp': marker_timestamps,
                     'marker_value': marker_values_raw
                 })
                 processed_stream_names.add(self._marker_stream_name)
                 self.logger.info(f"XDFReader - Loaded {len(xdf_markers_df)} raw XDF markers.")
            else:
                self.logger.warning(f"XDFReader - Marker stream '{self._marker_stream_name}' not found.")
            loaded_data['xdf_markers_df'] = xdf_markers_df

            # --- Load any other streams not explicitly handled by configured names ---
            self.logger.info("XDFReader - Checking for additional unconfigured streams.")
            for stream_name_key, stream_content_val in stream_map.items():
                if stream_name_key not in processed_stream_names:
                    self.logger.info(f"XDFReader - Loading additional stream: '{stream_name_key}' (type: {stream_content_val['info'].get('type', ['Unknown'])[0]})")
                    try:
                        # Store the essential parts of the stream directly
                        # For simplicity, we'll store the time_series as is.
                        # More sophisticated handling could try to infer channel orientation or type.
                        loaded_data[stream_name_key] = {
                            'time_series': np.array(stream_content_val['time_series']),
                            'time_stamps': np.array(stream_content_val['time_stamps']),
                            'info': stream_content_val['info'] # Full info dict for this stream
                        }
                        self.logger.info(f"XDFReader - Added additional stream '{stream_name_key}' to loaded_data.")
                    except Exception as e_other:
                        self.logger.error(f"XDFReader - Error processing additional stream '{stream_name_key}': {e_other}", exc_info=True)
                else:
                    self.logger.debug(f"XDFReader - Stream '{stream_name_key}' was already processed as a primary stream.")


        except Exception as e:
            self.logger.error(f"XDFReader - Critical error loading or parsing XDF file {xdf_file_path}: {e}", exc_info=True)
            return {}

        self.logger.info(f"XDFReader - Finished loading data for participant {participant_id}.")
        return loaded_data