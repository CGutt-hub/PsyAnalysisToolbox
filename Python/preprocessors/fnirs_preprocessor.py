import mne
import os
from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_long_channels, get_short_channels
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List # Added Optional, Tuple, List

class FNIRSPreprocessor:
    # Class-level defaults
    DEFAULT_BEER_LAMBERT_REMOVE_OD = True # Note: This specific flag is not directly used by MNE's core conversion functions in this sequence.
    DEFAULT_FILTER_H_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_L_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSPreprocessor initialized.")

    def process(self, fnirs_raw_od, participant_id: str, output_dir: str,
                  beer_lambert_ppf_config: Union[float, Tuple[float, float]],
                  short_channel_regression_config, # Moved up - non-default
                  motion_correction_method_config, # Moved up - non-default
                  filter_band_config: Tuple[Optional[float], Optional[float]], # Moved up - non-default
                  # Optional parameters that were previously hardcoded to defaults
                  beer_lambert_remove_od_config: Optional[bool] = None,
                  filter_h_trans_bandwidth_config: Optional[Union[str, float]] = None,
                  filter_l_trans_bandwidth_config: Optional[Union[str, float]] = None,
                  filter_fir_design_config: Optional[str] = None):
        """
        Preprocesses raw fNIRS optical density data to haemoglobin concentration.
        Args:
            fnirs_raw_od (mne.io.Raw): Raw fNIRS optical density data.
            beer_lambert_ppf_config (Union[float, Tuple[float, float]]): Partial pathlength factor(s).
            short_channel_regression_config (bool): Whether to apply short-channel regression.
            motion_correction_method_config (str): Method for motion correction (e.g., 'tddr', 'none').
            filter_band_config (tuple): Low and high cut-off frequencies for filtering (e.g., (0.01, 0.1)).
            beer_lambert_remove_od_config (Optional[bool]): Conceptual flag. Defaults to class default.
            filter_h_trans_bandwidth_config (Optional[Union[str, float]]): High-pass transition bandwidth. Defaults to class default.
            filter_l_trans_bandwidth_config (Optional[Union[str, float]]): Low-pass transition bandwidth. Defaults to class default.
            filter_fir_design_config (Optional[str]): FIR filter design. Defaults to class default.
        Returns:
            mne.io.Raw: Preprocessed fNIRS data containing haemoglobin (HbO, HbR) concentrations,
                        or None if an error occurs.
        """
        if fnirs_raw_od is None or not participant_id or not output_dir:
            self.logger.warning("FNIRSPreprocessor - No raw fNIRS OD data provided. Skipping.")
            return None
        
        # Validate required configurations
        if beer_lambert_ppf_config is None:
             self.logger.error("FNIRSPreprocessor - 'beer_lambert_ppf_config' is required but not provided. Skipping.")
             return None
        if not (isinstance(beer_lambert_ppf_config, (float, int)) or \
                (isinstance(beer_lambert_ppf_config, tuple) and len(beer_lambert_ppf_config) == 2 and \
                 all(isinstance(x, (float, int)) for x in beer_lambert_ppf_config))):
            self.logger.error(f"FNIRSPreprocessor - 'beer_lambert_ppf_config' must be a float or a tuple of two floats. Got {type(beer_lambert_ppf_config)}. Skipping.")
            return None

        if filter_band_config is None or not (isinstance(filter_band_config, (tuple, list)) and len(filter_band_config) == 2):
            self.logger.error("FNIRSPreprocessor - 'filter_band_config' must be a tuple or list of two floats (low, high). Skipping.")
            return None

        # Determine final config values, using defaults if not provided or invalid
        final_beer_lambert_remove_od = self.DEFAULT_BEER_LAMBERT_REMOVE_OD
        if beer_lambert_remove_od_config is not None:
            if isinstance(beer_lambert_remove_od_config, bool):
                final_beer_lambert_remove_od = beer_lambert_remove_od_config
            else:
                self.logger.warning(f"FNIRSPreprocessor: Invalid 'beer_lambert_remove_od_config' ('{beer_lambert_remove_od_config}'). Using default: {self.DEFAULT_BEER_LAMBERT_REMOVE_OD}.")

        final_filter_h_trans_bandwidth = self.DEFAULT_FILTER_H_TRANS_BANDWIDTH
        if filter_h_trans_bandwidth_config is not None:
            if isinstance(filter_h_trans_bandwidth_config, (str, float, int)): # Allow int for convenience
                final_filter_h_trans_bandwidth = filter_h_trans_bandwidth_config
            else:
                self.logger.warning(f"FNIRSPreprocessor: Invalid 'filter_h_trans_bandwidth_config' ('{filter_h_trans_bandwidth_config}'). Using default: {self.DEFAULT_FILTER_H_TRANS_BANDWIDTH}.")

        final_filter_l_trans_bandwidth = self.DEFAULT_FILTER_L_TRANS_BANDWIDTH
        if filter_l_trans_bandwidth_config is not None:
            if isinstance(filter_l_trans_bandwidth_config, (str, float, int)): # Allow int for convenience
                final_filter_l_trans_bandwidth = filter_l_trans_bandwidth_config
            else:
                self.logger.warning(f"FNIRSPreprocessor: Invalid 'filter_l_trans_bandwidth_config' ('{filter_l_trans_bandwidth_config}'). Using default: {self.DEFAULT_FILTER_L_TRANS_BANDWIDTH}.")

        final_filter_fir_design = self.DEFAULT_FILTER_FIR_DESIGN
        if filter_fir_design_config is not None:
            if isinstance(filter_fir_design_config, str) and filter_fir_design_config.strip():
                final_filter_fir_design = filter_fir_design_config.strip()
            else:
                self.logger.warning(f"FNIRSPreprocessor: Invalid 'filter_fir_design_config' ('{filter_fir_design_config}'). Using default: {self.DEFAULT_FILTER_FIR_DESIGN}.")

        # short_channel_regression_config and motion_correction_method_config are used directly.
        # filter_band_config is validated earlier.

        self.logger.info(f"FNIRSPreprocessor - Starting fNIRS preprocessing with effective configs: "
                         f"BeerLambertPPF={beer_lambert_ppf_config}, RemoveOD={final_beer_lambert_remove_od}, "
                         f"ShortChannelRegression={short_channel_regression_config}, "
                         f"MotionCorrectionMethod='{motion_correction_method_config}', "
                         f"FilterBand={filter_band_config}, "
                         f"FilterHTBW='{final_filter_h_trans_bandwidth}', FilterLTBW='{final_filter_l_trans_bandwidth}', "
                         f"FilterFIRDesign='{final_filter_fir_design}'.")

        try:
            # Ensure data is loaded
            if hasattr(fnirs_raw_od, '_data') and fnirs_raw_od._data is None and fnirs_raw_od.preload is False:
                fnirs_raw_od.load_data(verbose=False)

            # Verify that raw data contains CW amplitude data
            if 'fnirs_cw_amplitude' not in fnirs_raw_od.info.get_channel_types():
                self.logger.error("FNIRSPreprocessor - No continuous wave amplitude fNIRS data found. Please ensure the data is correctly loaded.")
                return None

            self.logger.info(f"FNIRSPreprocessor - Applying Beer-Lambert Law (PPF={beer_lambert_ppf_config}).")
            # Assuming fnirs_raw_od is raw intensity data, first convert to OD
            raw_od = mne.preprocessing.nirs.optical_density(fnirs_raw_od.copy())
            # Then convert OD to haemoglobin concentration
            raw_haemo = raw_od.to_concentration(ppf=beer_lambert_ppf_config)
            
            # Check for widespread NaNs/Infs immediately after Beer-Lambert conversion
            raw_haemo_data_check = raw_haemo.get_data()
            if np.all(np.isnan(raw_haemo_data_check)) or np.all(np.isinf(raw_haemo_data_check)):
                self.logger.error("FNIRSPreprocessor - Data is all NaN/Inf after Beer-Lambert law. Cannot proceed.")
                return None
            elif np.any(np.isnan(raw_haemo_data_check)) or np.any(np.isinf(raw_haemo_data_check)):
                 self.logger.warning("FNIRSPreprocessor - NaN/Inf values detected after Beer-Lambert law. Subsequent cleaning will attempt to address this.")
            del raw_haemo_data_check # Free memory
            
            if short_channel_regression_config:
                try:
                    # Check if there are any short channels defined
                    short_chs = get_short_channels(raw_haemo)
                    long_chs = get_long_channels(raw_haemo)
                    if not short_chs.empty and not long_chs.empty:
                        self.logger.info(f"FNIRSPreprocessor - Found {len(short_chs)} short channels. Applying short-channel regression.")
                        # Ensure raw_haemo includes both long and short for regression
                        raw_haemo_corrected_short = short_channel_regression(raw_haemo.copy())
                        raw_haemo = raw_haemo_corrected_short
                    else:
                        self.logger.info("FNIRSPreprocessor - No short channels found or defined. Skipping short-channel regression.")
                except Exception as e_scr:
                    self.logger.warning(f"FNIRSPreprocessor - Short-channel regression failed: {e_scr}. Continuing without it.", exc_info=True)
            
            if motion_correction_method_config == 'tddr':
                self.logger.info("FNIRSPreprocessor - Applying TDDR motion artifact correction.")
                # TDDR might be sensitive to NaNs or Infs if present
                current_data = raw_haemo.get_data()
                if np.any(np.isnan(current_data)) or np.any(np.isinf(current_data)):
                    self.logger.warning("FNIRSPreprocessor - NaN or Inf values found in data before TDDR. Attempting to clean by channel-wise interpolation.")
                    
                    # Replace Infs with NaNs to allow pandas interpolation
                    current_data[np.isinf(current_data)] = np.nan # type: ignore
                    
                    # Define a robust interpolation function for apply_function
                    def interpolate_nan_channel_wise(channel_data):
                        s = pd.Series(channel_data)
                        # Interpolate NaNs
                        s_interpolated = s.interpolate(method='linear', limit_direction='both')
                        # Fill any remaining NaNs at the very start/end using ffill and bfill
                        s_filled_ends = s_interpolated.ffill().bfill()
                        # If channel was all NaNs (or became all NaNs), fill with 0
                        return s_filled_ends.fillna(0).to_numpy()

 # Apply the interpolation channel by channel
                    # Create a new Raw object with the cleaned data to avoid in-place modification issues with apply_function
                    cleaned_data = np.array([interpolate_nan_channel_wise(ch_data) for ch_data in current_data])
                    raw_haemo = mne.io.RawArray(cleaned_data, raw_haemo.info, first_samp=raw_haemo.first_samp, verbose=False)

                corrected_haemo = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
 # Add other motion correction methods like 'savgol' if needed
            # elif motion_correction_method_config == 'savgol':
            # self.logger.info("FNIRSPreprocessor - Applying Savitzky-Golay filter for motion artifact correction.")
            #     # Example: corrected_haemo = raw_haemo.copy().filter(..., method='savgol', h_freq=None, l_freq=None, **some_savgol_params_config)
            #     corrected_haemo = raw_haemo # Placeholder if savgol not fully implemented here            
            elif motion_correction_method_config and motion_correction_method_config.lower() != 'none':
                self.logger.warning(f"FNIRSPreprocessor - Unsupported motion correction method '{motion_correction_method_config}' specified. No motion correction applied beyond initial processing.")
                corrected_haemo = raw_haemo # Pass through
            else:
                self.logger.info(f"FNIRSPreprocessor - Motion correction method is '{motion_correction_method_config}'. No specific motion correction applied beyond initial processing.")
                corrected_haemo = raw_haemo # Pass through if no method or 'none'
            self.logger.info(f"FNIRSPreprocessor - Applying band-pass filter ({filter_band_config[0]}-{filter_band_config[1]} Hz).")
            corrected_haemo.filter(l_freq=filter_band_config[0], h_freq=filter_band_config[1],
                                   h_trans_bandwidth=final_filter_h_trans_bandwidth,
                                   l_trans_bandwidth=final_filter_l_trans_bandwidth,
 fir_design=final_filter_fir_design, verbose=False)
            # Convert MNE object to DataFrame
            df = self._convert_haemo_to_dataframe(corrected_haemo)
            self.logger.info("FNIRSPreprocessor - fNIRS preprocessing completed.")
            return df
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error during fNIRS preprocessing: {e}", exc_info=True)
            return None

    def save_preprocessed_data(self, fnirs_haemo_obj, participant_id, output_dir):
        """Saves the preprocessed fNIRS haemoglobin data to a CSV file."""
        if fnirs_haemo_obj is None:
            self.logger.warning("FNIRSPreprocessor - No processed fNIRS object to save (CSV).")
            return

            
    def _convert_haemo_to_dataframe(self, fnirs_haemo_obj):
        """Converts MNE Raw object to a DataFrame."""
        if fnirs_haemo_obj is None:
            self.logger.warning("FNIRSPreprocessor - No processed fNIRS object to convert.")
            return pd.DataFrame()
        try:
            data = fnirs_haemo_obj.get_data()
            ch_names = fnirs_haemo_obj.ch_names
            df = pd.DataFrame(data.T, columns=ch_names)  # Transpose to have channels as columns

            # Add a time column (in seconds, relative to the start)
            times = fnirs_haemo_obj.times
            df['time'] = times  # Add as a new column

            # Reorder columns to have 'time' as the first column
            cols = ['time'] + ch_names
            df = df[cols]

            return df
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error converting MNE object to DataFrame: {e}", exc_info=True)
            return pd.DataFrame()