"""
Data Conversion Utility Module
-----------------------------
Provides helpers for converting between data formats (e.g., DataFrame <-> MNE Raw).
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any, Dict
import numpy as np
import mne

# Add conversion functions here as needed, with full docstrings and type hints.

# Placeholder example

def _create_eeg_mne_raw_from_df(eeg_df: Any, config: Any, logger: logging.Logger) -> Dict[str, Any]:
    """
    Converts an EEG DataFrame to an MNE Raw object.
    Filters out non-EEG columns and uses config for channel info.
    """
    logger.info("DataConversion: Creating MNE Raw from EEG DataFrame.")
    if eeg_df is None or len(eeg_df) == 0:
        logger.error("EEG DataFrame is empty or None.")
        return {}
    # Identify columns to exclude
    exclude_cols = ['TriggerStream', 'time_xdf', 'time_sec']
    eeg_ch_names = [col for col in eeg_df.columns if col not in exclude_cols]
    if not eeg_ch_names:
        logger.error("No EEG channels found after filtering auxiliary columns.")
        return {}
    logger.info(f"EEG channels used for MNE conversion: {eeg_ch_names}")
    # Extract EEG data
    eeg_data = eeg_df[eeg_ch_names].to_numpy().T  # shape: (n_channels, n_times)
    # Get sampling frequency from config
    sfreq = float(config.get('EEG', 'resample_sfreq', fallback=250.0))
    # Create MNE info
    info = mne.create_info(
        ch_names=eeg_ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    # Set montage if specified
    montage_name = config.get('EEG', 'montage_name', fallback=None)
    if montage_name:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            info.set_montage(montage)
            logger.info(f"Set montage: {montage_name}")
        except Exception as e:
            logger.warning(f"Could not set montage '{montage_name}': {e}")
    # Create RawArray
    try:
        raw = mne.io.RawArray(eeg_data, info)
        logger.info("Successfully created MNE Raw object from EEG DataFrame.")
        return {'eeg_mne_raw': raw}
    except Exception as e:
        logger.error(f"Failed to create MNE Raw object: {e}")
        return {}

def _create_fnirs_mne_raw_from_df(fnirs_df: Any, config: Any, logger: logging.Logger) -> Dict[str, Any]:
    """
    Converts an fNIRS DataFrame to an MNE Raw object.
    Filters out non-fNIRS columns and uses config for channel info.
    """
    logger.info("DataConversion: Creating MNE Raw from fNIRS DataFrame.")
    if fnirs_df is None or len(fnirs_df) == 0:
        logger.error("fNIRS DataFrame is empty or None.")
        return {}
    # Identify columns to exclude (customize as needed)
    exclude_cols = ['time_xdf', 'time_sec']
    fnirs_ch_names = [col for col in fnirs_df.columns if col not in exclude_cols]
    if not fnirs_ch_names:
        logger.error("No fNIRS channels found after filtering auxiliary columns.")
        return {}
    logger.info(f"fNIRS channels used for MNE conversion: {fnirs_ch_names}")
    # Extract fNIRS data
    fnirs_data = fnirs_df[fnirs_ch_names].to_numpy().T  # shape: (n_channels, n_times)
    # Get sampling frequency from config
    sfreq = float(config.get('FNIRS', 'sfreq', fallback=7.81))
    # Create MNE info
    info = mne.create_info(
        ch_names=fnirs_ch_names,
        sfreq=sfreq,
        ch_types='fnirs_cw_amplitude'
    )
    # Set montage if specified
    montage_name = config.get('FNIRS', 'montage_name', fallback=None)
    if montage_name:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            info.set_montage(montage)
            logger.info(f"Set fNIRS montage: {montage_name}")
        except Exception as e:
            logger.warning(f"Could not set fNIRS montage '{montage_name}': {e}")
    # Create RawArray
    try:
        raw = mne.io.RawArray(fnirs_data, info)
        logger.info("Successfully created MNE Raw object from fNIRS DataFrame.")
        return {'fnirs_mne_raw': raw}
    except Exception as e:
        logger.error(f"Failed to create MNE Raw object for fNIRS: {e}")
        return {} 