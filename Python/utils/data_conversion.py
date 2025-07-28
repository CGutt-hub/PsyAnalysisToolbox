import pandas as pd
import numpy as np
import mne
from mne.io import RawArray
from configparser import ConfigParser, NoOptionError, NoSectionError
from typing import Dict, Any, List

def _create_eeg_mne_raw_from_df(eeg_df: pd.DataFrame, config: ConfigParser, p_logger) -> Dict[str, Any] | None:
    """Converts a raw EEG DataFrame to an MNE RawArray object with proper channel typing and montage."""
    try:
        p_logger.info("Converting EEG DataFrame to MNE Raw object...")
        sfreq_eeg = 1 / np.mean(np.diff(eeg_df['time_sec']))
        p_logger.info(f"Calculated EEG sampling frequency: {sfreq_eeg:.2f} Hz")
        exclude_prefixes_str = config.get('ChannelManagement', 'eeg_channel_exclude_prefixes', fallback='')
        exclude_prefixes = tuple([p.strip() for p in exclude_prefixes_str.split(',') if p.strip()])
        aux_channels_str = config.get('ChannelManagement', 'eeg_aux_channel_names', fallback='')
        aux_channel_set = {ch.strip() for ch in aux_channels_str.split(',') if ch.strip()}
        trigger_channel_name = config.get('ChannelManagement', 'eeg_trigger_channel_name', fallback=None)
        if trigger_channel_name:
            trigger_channel_name = trigger_channel_name.strip()
        final_ch_names: List[str] = []
        final_ch_types: List[str] = []
        ch_types_map = {}
        for ch_name_raw in eeg_df.columns:
            ch_name = ch_name_raw.strip()
            if ch_name in ['time_xdf', 'time_sec']:
                continue
            if ch_name.startswith(exclude_prefixes):
                p_logger.debug(f"Excluding channel '{ch_name}' based on prefix.")
                continue
            if trigger_channel_name and ch_name == trigger_channel_name:
                final_ch_names.append(ch_name)
                final_ch_types.append('stim')
                ch_types_map[ch_name] = 'stim'
            elif ch_name in aux_channel_set:
                final_ch_names.append(ch_name)
                final_ch_types.append('misc')
                ch_types_map[ch_name] = 'misc'
            else:
                final_ch_names.append(ch_name)
                final_ch_types.append('eeg')
                ch_types_map[ch_name] = 'eeg'
        if not final_ch_names:
            p_logger.error("No valid EEG or auxiliary channels found after filtering. Cannot create MNE Raw object.")
            return None
        p_logger.info(f"Identified {final_ch_types.count('eeg')} EEG, {final_ch_types.count('stim')} stim, and {final_ch_types.count('misc')} auxiliary channels.")
        info = mne.create_info(ch_names=final_ch_names, sfreq=sfreq_eeg, ch_types=final_ch_types)  # type: ignore[arg-type]
        data_for_raw = eeg_df[final_ch_names].to_numpy().T
        raw_eeg = RawArray(data_for_raw, info)
        montage_name = config.get('EEG', 'montage_name', fallback=None)
        if montage_name:
            montage = mne.channels.make_standard_montage(montage_name)
            raw_eeg.set_montage(montage, on_missing='warn')
            p_logger.info(f"Applied '{montage_name}' montage to EEG data.")
        return {'eeg_mne_raw': raw_eeg, 'eeg_channel_names': final_ch_names, 'eeg_ch_types_map': ch_types_map}
    except Exception as e:
        p_logger.error(f"Failed during EEG MNE object creation: {e}", exc_info=True)
        return None

def _create_fnirs_mne_raw_from_df(fnirs_df: pd.DataFrame, config: ConfigParser, p_logger) -> Dict[str, Any] | None:
    """Converts a raw fNIRS DataFrame to an MNE RawArray object with MNE-compliant channel names."""
    try:
        p_logger.info("Converting fNIRS DataFrame to MNE Raw object...")
        sfreq_fnirs = 1 / np.mean(np.diff(fnirs_df['time_sec']))
        p_logger.info(f"Calculated fNIRS sampling frequency: {sfreq_fnirs:.2f} Hz")
        wavelengths_str = config.get('FNIRS', 'wavelengths')
        sd_pairs_str = config.get('FNIRS', 'sd_pairs_ordered')
        wavelengths = [int(w.strip()) for w in wavelengths_str.split(',')]
        sd_pairs = [p.strip() for p in sd_pairs_str.split(',')]
        mne_ch_names = [f"{sd.replace('-', '_')} {wl}" for wl in wavelengths for sd in sd_pairs]
        data_cols = [col for col in fnirs_df.columns if col not in ['time_xdf', 'time_sec']]
        num_expected_channels = len(mne_ch_names)
        num_actual_data_cols = len(data_cols)
        if num_actual_data_cols < num_expected_channels:
            p_logger.error(f"fNIRS channel mismatch: Config implies {num_expected_channels} channels, but data stream only has {num_actual_data_cols} data columns. Cannot proceed.")
            return None
        fnirs_data_cols = data_cols[:num_expected_channels]
        if num_actual_data_cols > num_expected_channels:
            p_logger.warning(f"fNIRS data stream has {num_actual_data_cols} columns, but config specifies {num_expected_channels} channels. Using the first {num_expected_channels} columns.")
        info = mne.create_info(ch_names=mne_ch_names, sfreq=sfreq_fnirs, ch_types='fnirs_cw_amplitude')
        raw_fnirs_amp = RawArray(fnirs_df[fnirs_data_cols].to_numpy().T, info)
        p_logger.info("Creating and applying fNIRS digital montage...")
        source_locs_str = config.get('FNIRS_Montage', 'source_locations')
        detector_locs_str = config.get('FNIRS_Montage', 'detector_locations')
        def parse_locs(loc_str):
            loc_dict = {}
            for item in loc_str.split(';'):
                label, coords = item.split(':')
                loc_dict[label.strip()] = tuple(map(float, coords.split(',')))
            return loc_dict
        dig_montage = mne.channels.make_dig_montage(
            ch_pos=parse_locs(source_locs_str) | parse_locs(detector_locs_str),
            lpa=[float(c.strip()) for c in config.get('FNIRS_Montage', 'lpa').split(',')],
            rpa=[float(c.strip()) for c in config.get('FNIRS_Montage', 'rpa').split(',')],
            nasion=[float(c.strip()) for c in config.get('FNIRS_Montage', 'nasion').split(',')],
            coord_frame='head'
        )
        raw_fnirs_amp.set_montage(dig_montage, on_missing='warn')
        p_logger.info("Successfully applied fNIRS montage.")
        n_wavelengths = len(wavelengths)
        n_channels = len(raw_fnirs_amp.ch_names)
        if n_channels % n_wavelengths != 0:
            p_logger.error(f"fNIRS channel count ({n_channels}) is not a multiple of number of wavelengths ({n_wavelengths}). Cannot assign wavelengths correctly.")
            return None
        n_pairs = n_channels // n_wavelengths
        for i, ch_name in enumerate(raw_fnirs_amp.ch_names):
            if n_channels == n_pairs * n_wavelengths and n_pairs == len(sd_pairs):
                wl_idx = i // n_pairs
                if wl_idx >= n_wavelengths:
                    wl_idx = n_wavelengths - 1
                wavelength = wavelengths[wl_idx]
            else:
                wavelength = wavelengths[i % n_wavelengths]
            raw_fnirs_amp.info['chs'][i]['loc'][9] = wavelength
        p_logger.info(f"Successfully embedded wavelength data into fNIRS MNE object. Mapping: {[raw_fnirs_amp.info['chs'][i]['loc'][9] for i in range(n_channels)]}")
        return {'fnirs_mne_raw': raw_fnirs_amp, 'fnirs_channel_names': mne_ch_names}
    except (NoSectionError, NoOptionError) as e:
        p_logger.error(f"Missing fNIRS channel configuration in [FNIRS] section: {e}. Required for MNE object creation.")
        return None
    except Exception as e:
        p_logger.error(f"Failed during fNIRS MNE object creation: {e}", exc_info=True)
        return None 