import mne
import os
from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_long_channels, get_short_channels
import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Tuple, List, Dict, Any # Added Dict for type hints

class FNIRSPreprocessor:
    # Class-level defaults
    DEFAULT_BEER_LAMBERT_REMOVE_OD = True # Note: This specific flag is not directly used by MNE's core conversion functions in this sequence.
    DEFAULT_FILTER_H_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_L_TRANS_BANDWIDTH: Union[str, float] = 'auto'
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("FNIRSPreprocessor initialized.")

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validates the provided fNIRS configuration dictionary."""
        ppf_config = config.get('beer_lambert_ppf')
        if ppf_config is None:
            self.logger.error("FNIRSPreprocessor - 'beer_lambert_ppf' is required but not provided. Skipping.")
            return False
        if not (isinstance(ppf_config, (float, int)) or \
                (isinstance(ppf_config, tuple) and len(ppf_config) == 2 and \
                 all(isinstance(x, (float, int)) for x in ppf_config))):
            self.logger.error(f"FNIRSPreprocessor - 'beer_lambert_ppf' must be a float or a tuple of two floats. Got {type(ppf_config)}. Skipping.")
            return False

        filter_band = config.get('filter_band')
        if not (isinstance(filter_band, (tuple, list)) and len(filter_band) == 2):
            self.logger.error("FNIRSPreprocessor - 'filter_band' must be a tuple or list of two floats (low, high). Skipping.")
            return False
        return True

    def _resolve_final_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves final configuration values, applying defaults for optional parameters."""
        resolved = config.copy()
        resolved.setdefault('beer_lambert_remove_od', self.DEFAULT_BEER_LAMBERT_REMOVE_OD)
        resolved.setdefault('filter_h_trans_bandwidth', self.DEFAULT_FILTER_H_TRANS_BANDWIDTH)
        resolved.setdefault('filter_l_trans_bandwidth', self.DEFAULT_FILTER_L_TRANS_BANDWIDTH)
        resolved.setdefault('filter_fir_design', self.DEFAULT_FILTER_FIR_DESIGN)
        return resolved

    def _apply_beer_lambert(self, raw_intensity: mne.io.Raw, ppf: Union[float, Tuple[float, float]]) -> Optional[mne.io.Raw]:
        """Converts intensity data to optical density, then to haemoglobin concentration."""
        self.logger.info(f"FNIRSPreprocessor - Applying Beer-Lambert Law (PPF={ppf}).")
        raw_od = mne.preprocessing.nirs.optical_density(raw_intensity.copy())
        raw_haemo = raw_od.to_concentration(ppf=ppf)
        
        haemo_data = raw_haemo.get_data()
        if np.all(np.isnan(haemo_data)) or np.all(np.isinf(haemo_data)):
            self.logger.error("FNIRSPreprocessor - Data is all NaN/Inf after Beer-Lambert law. Cannot proceed.")
            return None
        elif np.any(np.isnan(haemo_data)) or np.any(np.isinf(haemo_data)):
            self.logger.warning("FNIRSPreprocessor - NaN/Inf values detected after Beer-Lambert law. Subsequent cleaning will attempt to address this.")
        return raw_haemo

    def _apply_short_channel_regression(self, raw_haemo: mne.io.Raw) -> mne.io.Raw:
        """Applies short-channel regression if short channels are present."""
        try:
            short_chs = get_short_channels(raw_haemo.info)
            if not short_chs.empty:
                self.logger.info(f"FNIRSPreprocessor - Found {len(short_chs)} short channels. Applying short-channel regression.")
                return short_channel_regression(raw_haemo.copy())
            else:
                self.logger.info("FNIRSPreprocessor - No short channels found. Skipping short-channel regression.")
                return raw_haemo
        except Exception as e:
            self.logger.warning(f"FNIRSPreprocessor - Short-channel regression failed: {e}. Continuing without it.", exc_info=True)
            return raw_haemo

    def _apply_motion_correction(self, raw_haemo: mne.io.Raw, method: str) -> mne.io.Raw:
        """Applies the specified motion correction technique."""
        if method.lower() == 'tddr':
            self.logger.info("FNIRSPreprocessor - Applying TDDR motion artifact correction.")
            current_data = raw_haemo.get_data()
            if np.any(np.isnan(current_data)) or np.any(np.isinf(current_data)):
                self.logger.warning("FNIRSPreprocessor - NaN or Inf values found before TDDR. Attempting channel-wise interpolation.")
                current_data = np.where(np.isinf(current_data), np.nan, current_data)
                
                def interpolate_nan_channel_wise(channel_data):
                    s = pd.Series(channel_data)
                    s_interpolated = s.interpolate(method='linear', limit_direction='both')
                    s_filled = s_interpolated.ffill().bfill()
                    return s_filled.fillna(0).to_numpy()

                cleaned_data = np.array([interpolate_nan_channel_wise(ch_data) for ch_data in current_data])
                raw_haemo = mne.io.RawArray(cleaned_data, raw_haemo.info, first_samp=raw_haemo.first_samp, verbose=False) # type: ignore
            
            return mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
        elif method.lower() != 'none':
            self.logger.warning(f"FNIRSPreprocessor - Unsupported motion correction method '{method}'. No correction applied.")
        else:
            self.logger.info("FNIRSPreprocessor - No motion correction method specified.")
        return raw_haemo

    def _apply_filter(self, raw_haemo: mne.io.Raw, config: Dict[str, Any]) -> mne.io.Raw:
        """Applies a band-pass filter to the data."""
        l_freq, h_freq = config['filter_band']
        self.logger.info(f"FNIRSPreprocessor - Applying band-pass filter ({l_freq}-{h_freq} Hz).")
        raw_haemo.filter(l_freq=l_freq, h_freq=h_freq,
                         h_trans_bandwidth=config['filter_h_trans_bandwidth'],
                         l_trans_bandwidth=config['filter_l_trans_bandwidth'],
                         fir_design=config['filter_fir_design'], verbose=False)
        return raw_haemo

    def process(self, fnirs_raw_intensity: mne.io.Raw, config: Dict[str, Any]) -> Optional[mne.io.Raw]:
        """
        Preprocesses raw fNIRS optical density data to haemoglobin concentration.
        Args:
            fnirs_raw_intensity (mne.io.Raw): Raw fNIRS intensity data (fnirs_cw_amplitude).
            config (Dict[str, Any]): Dictionary with all preprocessing parameters.
        Returns:
            mne.io.Raw: Preprocessed fNIRS data containing haemoglobin (HbO, HbR) concentrations,
                        or None if an error occurs.
        """
        if fnirs_raw_intensity is None:
            self.logger.warning("FNIRSPreprocessor - No raw fNIRS intensity data provided. Skipping.")
            return None
        
        if not self._validate_config(config):
            return None

        final_config = self._resolve_final_configs(config)
        self.logger.info(f"FNIRSPreprocessor - Starting fNIRS preprocessing with effective configs: {final_config}")

        try:
            # Ensure data is loaded
            if hasattr(fnirs_raw_intensity, '_data') and fnirs_raw_intensity._data is None and fnirs_raw_intensity.preload is False:
                fnirs_raw_intensity.load_data(verbose=False)

            # Verify that raw data contains CW amplitude data
            if 'fnirs_cw_amplitude' not in fnirs_raw_intensity.get_channel_types():
                self.logger.error("FNIRSPreprocessor - No continuous wave amplitude fNIRS data found. Please ensure the data is correctly loaded.")
                return None

            # --- Processing Pipeline ---
            raw_haemo = self._apply_beer_lambert(fnirs_raw_intensity, final_config['beer_lambert_ppf'])
            if raw_haemo is None:
                return None

            if final_config.get('short_channel_regression'):
                raw_haemo = self._apply_short_channel_regression(raw_haemo)

            raw_haemo = self._apply_motion_correction(raw_haemo, final_config.get('motion_correction_method', 'none'))
            
            corrected_haemo = self._apply_filter(raw_haemo, final_config)

            self.logger.info("FNIRSPreprocessor - fNIRS preprocessing completed.")
            return corrected_haemo
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error during fNIRS preprocessing: {e}", exc_info=True)
            return None