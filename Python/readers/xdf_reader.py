import os
import pyxdf
import mne
import numpy as np
import pandas as pd
from typing import Optional

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
            eeg_stream_name (Optional[str]): Name of the EEG stream. Defaults to XDFReader.DEFAULT_EEG_STREAM_NAME.
            fnirs_stream_name (Optional[str]): Name of the fNIRS stream. Defaults to XDFReader.DEFAULT_FNIRS_STREAM_NAME.
            ecg_stream_name (Optional[str]): Name of the ECG stream. Defaults to XDFReader.DEFAULT_ECG_STREAM_NAME.
            eda_stream_name (Optional[str]): Name of the EDA stream. Defaults to XDFReader.DEFAULT_EDA_STREAM_NAME.
            marker_stream_name (Optional[str]): Name of the Marker stream. Defaults to XDFReader.DEFAULT_MARKER_STREAM_NAME.
        """
        self.logger = logger

        # Configure EEG stream name
        self._eeg_stream_name = self.DEFAULT_EEG_STREAM_NAME
        if eeg_stream_name is not None:
            if isinstance(eeg_stream_name, str) and eeg_stream_name.strip():
                self._eeg_stream_name = eeg_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{eeg_stream_name}') provided for 'eeg_stream_name'. "
                    f"Using default: '{self.DEFAULT_EEG_STREAM_NAME}'."
                )

        # Configure fNIRS stream name
        self._fnirs_stream_name = self.DEFAULT_FNIRS_STREAM_NAME
        if fnirs_stream_name is not None:
            if isinstance(fnirs_stream_name, str) and fnirs_stream_name.strip():
                self._fnirs_stream_name = fnirs_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{fnirs_stream_name}') provided for 'fnirs_stream_name'. "
                    f"Using default: '{self.DEFAULT_FNIRS_STREAM_NAME}'."
                )

        # Configure ECG stream name
        self._ecg_stream_name = self.DEFAULT_ECG_STREAM_NAME
        if ecg_stream_name is not None:
            if isinstance(ecg_stream_name, str) and ecg_stream_name.strip():
                self._ecg_stream_name = ecg_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{ecg_stream_name}') provided for 'ecg_stream_name'. "
                    f"Using default: '{self.DEFAULT_ECG_STREAM_NAME}'."
                )

        # Configure EDA stream name
        self._eda_stream_name = self.DEFAULT_EDA_STREAM_NAME
        if eda_stream_name is not None:
            if isinstance(eda_stream_name, str) and eda_stream_name.strip():
                self._eda_stream_name = eda_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{eda_stream_name}') provided for 'eda_stream_name'. "
                    f"Using default: '{self.DEFAULT_EDA_STREAM_NAME}'."
                )

        # Configure Marker stream name
        self._marker_stream_name = self.DEFAULT_MARKER_STREAM_NAME
        if marker_stream_name is not None:
            if isinstance(marker_stream_name, str) and marker_stream_name.strip():
                self._marker_stream_name = marker_stream_name
            else:
                self.logger.warning(
                    f"XDFReader: Invalid value ('{marker_stream_name}') provided for 'marker_stream_name'. "
                    f"Using default: '{self.DEFAULT_MARKER_STREAM_NAME}'."
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
                    try:
                        sfreq = float(stream['info']['nominal_srate'][0])
                        if sfreq <= 0: raise ValueError("Sampling rate must be positive.")
                    except (IndexError, TypeError, ValueError) as e_sfreq:
                        self.logger.error(f"XDFReader - Invalid sampling frequency for EEG stream '{self._eeg_stream_name}': {e_sfreq}. Skipping stream.")
                        raise # Re-raise to be caught by the outer try-except for this stream
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['eeg'] * len(ch_names)
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) # type: ignore
                    raw_eeg = mne.io.RawArray(data, info, verbose=False)

                    loaded_data['eeg'] = raw_eeg
                    #processed_stream_names.add(self._eeg_stream_name) # Will be processed later
                    self.logger.info(f"XDFReader - Loaded EEG data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"XDFReader - Error loading or processing EEG stream: {e}", exc_info=True)
                    loaded_data['eeg'] = None
            else:
                self.logger.warning(f"XDFReader - EEG stream '{self._eeg_stream_name}' not found.")
                loaded_data['eeg'] = None

            # Convert EEG to DataFrame if loaded
            if 'eeg' in loaded_data and loaded_data['eeg'] is not None:
                try:
                    raw_eeg = loaded_data['eeg']
                    eeg_data = raw_eeg.get_data()  # NumPy array: (channels, times)
                    eeg_times = raw_eeg.times  # NumPy array: (times,) in seconds relative to start
                    eeg_ch_names = raw_eeg.ch_names
                    eeg_sfreq = raw_eeg.info['sfreq']

                    eeg_df = pd.DataFrame(np.atleast_2d(eeg_data).T, columns=eeg_ch_names) # Ensure 2D before transpose
                    eeg_df['time'] = pd.Series(eeg_times) # Add a time column in seconds relative to the stream start.
                    eeg_df['time_xdf'] = loaded_data.get('eeg_times', np.full(len(eeg_times), np.nan)).tolist() # If ecg_times available, use them, else NaNs.

                    loaded_data['eeg_df'] = eeg_df  # Replace the MNE object with the DataFrame.
                    del loaded_data['eeg']  # Remove the original MNE object
                    self.logger.info(f"XDFReader - Converted EEG data to DataFrame with shape {eeg_df.shape}.")
                except Exception as e_convert:
                    self.logger.error(f"XDFReader - Error converting EEG to DataFrame: {e_convert}", exc_info=True)
                    # Even if conversion fails, leave 'eeg' as None.
                loaded_data['eeg'] = None

            # --- Load ECG ---
            if self._ecg_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading ECG stream: {self._ecg_stream_name}")
                stream = stream_map[self._ecg_stream_name]
                try:
                    # Assuming ECG is a single channel time series
                    ecg_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    try:
                        ecg_sfreq = float(stream['info']['nominal_srate'][0])
                        if ecg_sfreq <= 0: raise ValueError("Sampling rate must be positive.")
                    except (IndexError, TypeError, ValueError) as e_sfreq:
                        self.logger.error(f"XDFReader - Invalid sampling frequency for ECG stream '{self._ecg_stream_name}': {e_sfreq}. Skipping stream.")
                        raise
                    ecg_times = stream['time_stamps'] # Absolute timestamps from XDF

                    loaded_data['ecg_signal'] = ecg_signal
                    loaded_data['ecg_sfreq'] = ecg_sfreq
                    loaded_data['ecg_times'] = ecg_times # Store absolute times for alignment
                    #processed_stream_names.add(self._ecg_stream_name) # Processed later.
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

            # Convert ECG to DataFrame
            if 'ecg_signal' in loaded_data and loaded_data['ecg_signal'] is not None:
                try:
                    ecg_df = pd.DataFrame({'ecg_signal': loaded_data['ecg_signal']})
                    ecg_df['time_xdf'] = loaded_data.get('ecg_times', np.full(len(ecg_df), np.nan)) # If absolute times exist, use, else nan
                    if loaded_data['ecg_sfreq'] is not None:
                        ecg_df.index = pd.Index(np.arange(len(ecg_df)) / float(loaded_data['ecg_sfreq']), name='time') # relative time in seconds
                    else:
                        self.logger.warning("XDFReader - ECG sampling frequency is None, cannot set DataFrame index to relative time.")

                    loaded_data['ecg_df'] = ecg_df # Replace signal and times with DataFrame
                    del loaded_data['ecg_signal'], loaded_data['ecg_sfreq'], loaded_data['ecg_times']
                    self.logger.info(f"XDFReader - Converted ECG data to DataFrame with shape {ecg_df.shape}.")
                except Exception as e_convert:
                    self.logger.error(f"XDFReader - Error converting ECG to DataFrame: {e_convert}", exc_info=True)
                    # In case of error, we do NOT re-populate the old variables. The modality will be considered missing
                loaded_data['ecg_signal'] = None
                loaded_data['ecg_sfreq'] = None
                loaded_data['ecg_times'] = None

                loaded_data['ecg_times'] = None


            # --- Load EDA ---
            if self._eda_stream_name in stream_map:
                self.logger.info(f"XDFReader - Loading EDA stream: {self._eda_stream_name}")
                stream = stream_map[self._eda_stream_name]
                try:
                    # Assuming EDA is a single channel time series
                    eda_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    try:
                        eda_sfreq = float(stream['info']['nominal_srate'][0])
                        if eda_sfreq <= 0: raise ValueError("Sampling rate must be positive.")
                    except (IndexError, TypeError, ValueError) as e_sfreq:
                        self.logger.error(f"XDFReader - Invalid sampling frequency for EDA stream '{self._eda_stream_name}': {e_sfreq}. Skipping stream.")
                        raise
                    eda_times = stream['time_stamps'] # Absolute timestamps from XDF

                    loaded_data['eda_signal'] = eda_signal
                    loaded_data['eda_sfreq'] = eda_sfreq
                    loaded_data['eda_times'] = eda_times # Store absolute times for alignment
                    #processed_stream_names.add(self._eda_stream_name) # Processed later
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

            # Convert EDA data to DataFrame
            if 'eda_signal' in loaded_data and loaded_data['eda_signal'] is not None:
                try:
                    eda_df = pd.DataFrame({'eda_signal': loaded_data['eda_signal']})
                    eda_df['time_xdf'] = loaded_data.get('eda_times', np.full(len(eda_df), np.nan))  # If absolute times, else NaN
                    if loaded_data['eda_sfreq'] is not None:
                        eda_df.index = pd.Index(np.arange(len(eda_df)) / float(loaded_data['eda_sfreq']), name='time') # relative time
                    else:
                        self.logger.warning("XDFReader - EDA sampling frequency is None, cannot set DataFrame index to relative time.")
                        eda_df.index = pd.Index(np.arange(len(eda_df)), name='sample') # fallback to sample index
                    
                    loaded_data['eda_df'] = eda_df # Replace the signal and sfreq with the DataFrame
                    del loaded_data['eda_signal'], loaded_data['eda_sfreq'], loaded_data['eda_times']
                    self.logger.info(f"XDFReader - Converted EDA data to DataFrame with shape {eda_df.shape}.")
                except Exception as e_convert:
                    self.logger.error(f"XDFReader - Error converting EDA to DataFrame: {e_convert}", exc_info=True)
                    # In error, do NOT keep old variables, effectively marking the modality as missing.

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
                    try:
                        sfreq = float(stream['info']['nominal_srate'][0])
                        if sfreq <= 0: raise ValueError("Sampling rate must be positive.")
                    except (IndexError, TypeError, ValueError) as e_sfreq:
                        self.logger.error(f"XDFReader - Invalid sampling frequency for fNIRS stream '{self._fnirs_stream_name}': {e_sfreq}. Skipping stream.")
                        raise
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['fnirs_od'] * len(ch_names)

                    # MNE expects ch_types as a list of strings, one for each channel
                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) # type: ignore
                    raw_fnirs_od = mne.io.RawArray(data, info, verbose=False)
                    self.logger.info(f"XDFReader - Loaded fNIRS data with {len(ch_names)} channels at {sfreq} Hz.")
                    #processed_stream_names.add(self._fnirs_stream_name) # Will be processed into df later

                except Exception as e:
                    raw_fnirs_od = None # Already initialized, but good to be explicit on error path
                    fnirs_stream_start_time_xdf = None # Already initialized
                    self.logger.error(f"XDFReader - Error loading or processing fNIRS stream: {e}", exc_info=True)
            else:
                self.logger.warning(f"XDFReader - fNIRS stream '{self._fnirs_stream_name}' not found.")
                # raw_fnirs_od and fnirs_stream_start_time_xdf remain None as initialized

            # Convert fNIRS data to DataFrame
            if 'fnirs_od' in loaded_data and loaded_data['fnirs_od'] is not None:
                try:
                    raw_fnirs = loaded_data['fnirs_od']
                    fnirs_data = raw_fnirs.get_data()  # NumPy array: (channels, times)
                    fnirs_times = raw_fnirs.times  # NumPy array: (times,) in seconds
                    fnirs_ch_names = raw_fnirs.ch_names
                    fnirs_sfreq = raw_fnirs.info['sfreq']

                    fnirs_df = pd.DataFrame(fnirs_data.T, columns=fnirs_ch_names) # Transpose: (times, channels)
                    fnirs_df['time'] = fnirs_times # Time in seconds relative to the start

                    loaded_data['fnirs_od_df'] = fnirs_df  # Store as new DataFrame
                    del loaded_data['fnirs_od']  # Remove MNE Raw object
                    self.logger.info(f"XDFReader - Converted fNIRS data to DataFrame with shape {fnirs_df.shape}.")
                except Exception as e_convert:
                    self.logger.error(f"XDFReader - Error converting fNIRS data to DataFrame: {e_convert}", exc_info=True)
                    # On error, leave the 'fnirs_od' entry empty.
                loaded_data['fnirs_od'] = None

            loaded_data['fnirs_od'] = raw_fnirs_od
            loaded_data['fnirs_stream_start_time_xdf'] = fnirs_stream_start_time_xdf

            # --- Load Raw XDF Markers ---
            xdf_markers_df = pd.DataFrame() # Initialize in case marker stream is not found
            if self._marker_stream_name in stream_map:
                 marker_stream = stream_map[self._marker_stream_name]
                 marker_timestamps = marker_stream['time_stamps'] # Absolute XDF times
                 marker_values_raw = []
                 for val_list in marker_stream['time_series']:
                     if val_list and isinstance(val_list, (list, tuple)) and len(val_list) > 0:
                         marker_values_raw.append(val_list[0])
                     elif isinstance(val_list, (str, int, float)): # Handle scalar marker values directly
                         marker_values_raw.append(val_list)
                     else:
                         self.logger.warning(f"XDFReader - Unexpected marker value format: {val_list}. Appending as None.")
                         marker_values_raw.append(None) # Or skip, or use a placeholder

                 xdf_markers_df = pd.DataFrame({
                     'timestamp': marker_timestamps,
                     'marker_value': marker_values_raw
                 })
                 #processed_stream_names.add(self._marker_stream_name) # Is already a dataframe, will be treated as "other" stream if needed
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
