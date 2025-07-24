import pandas as pd

# Preprocessing helper functions moved from EV_orchestrator.py

def _preprocess_eeg_stream(config, components, p_logger, streams, pid, output_dir):
    """Helper to preprocess the EEG stream."""
    p_logger.info("Preprocessing EEG stream...")
    processed_artifacts = {}
    try:
        raw_eeg_initial = streams['eeg'] # The raw object before processing
        # --- Diagnostic Plot Generation ---
        fai_channels_to_inspect = ['F3', 'F4', 'Fp1', 'Fp2']
        channels_to_plot = [ch for ch in fai_channels_to_inspect if ch in raw_eeg_initial.ch_names]
        if channels_to_plot:
            p_logger.info(f"Generating diagnostic plot for raw FAI channels: {channels_to_plot}")
            try:
                import os
                import matplotlib
                diag_plot_dir = os.path.join(output_dir, "EEG_Preproc_Diagnostics")
                os.makedirs(diag_plot_dir, exist_ok=True)
                fig = raw_eeg_initial.plot(
                    picks=channels_to_plot,
                    duration=30,
                    start=0,
                    n_channels=len(channels_to_plot),
                    scalings='auto',
                    show=False,
                    title=f"Raw Data for FAI Channels - {pid}"
                )
                plot_path = os.path.join(diag_plot_dir, f"{pid}_raw_fai_channels_diagnostic.png")
                fig.savefig(plot_path, dpi=150) # type: ignore
                matplotlib.pyplot.close(fig) # type: ignore
                p_logger.info(f"Saved raw data diagnostic plot to: {plot_path}")
            except Exception as plot_e:
                p_logger.warning(f"Could not generate diagnostic plot for raw FAI channels: {plot_e}")
        eeg_processing_config = {
            'eeg_filter_band': (
                config.getfloat('EEG', 'eeg_filter_l_freq', fallback=None),
                config.getfloat('EEG', 'eeg_filter_h_freq', fallback=None)
            ),
            'ica_n_components': config.get('EEG', 'ica_n_components'),
            'ica_random_state': config.getint('EEG', 'ica_random_state', fallback=None),
            'ica_reject_threshold': config.getfloat('EEG', 'ica_reject_threshold'),
            'ica_method': config.get('EEG', 'ica_method', fallback='fastica'),
            'ica_extended': config.getboolean('EEG', 'ica_extended', fallback=False),
            'ica_accept_labels': [l.strip() for l in config.get('EEG', 'ica_accept_labels').split(',')],
            'eeg_reference': config.get('EEG', 'eeg_reference', fallback=None),
            'eeg_reference_projection': config.getboolean('EEG', 'eeg_reference_projection', fallback=None),
            'filter_fir_design': config.get('EEG', 'filter_fir_design', fallback=None),
            'ica_max_iter': config.get('EEG', 'ica_max_iter', fallback=None),
            'resample_sfreq': config.getfloat('EEG', 'resample_sfreq', fallback=None),
            'ica_labeling_method': config.get('EEG', 'ica_labeling_method', fallback=None)
        }
        ica_n_str = eeg_processing_config['ica_n_components']
        if isinstance(ica_n_str, str):
            if ica_n_str.isdigit():
                eeg_processing_config['ica_n_components'] = int(ica_n_str)
            elif '.' in ica_n_str:
                try:
                    eeg_processing_config['ica_n_components'] = float(ica_n_str)
                except ValueError:
                    pass
        ica_iter_str = eeg_processing_config['ica_max_iter']
        if isinstance(ica_iter_str, str) and ica_iter_str.isdigit():
            eeg_processing_config['ica_max_iter'] = int(ica_iter_str)
        eeg_preproc_artifacts = components['eeg_preprocessor'].process(
            raw_eeg=streams['eeg'],
            eeg_config=eeg_processing_config)
        if eeg_preproc_artifacts is not None:
            processed_raw = eeg_preproc_artifacts.get('eeg_processed_raw')
            if processed_raw:
                processed_artifacts['eeg_sfreq'] = processed_raw.info['sfreq']
            processed_artifacts.update(eeg_preproc_artifacts)
            p_logger.info("EEG preprocessing completed.")
    except Exception as e:
        p_logger.error(f"An error occurred during EEG preprocessing: {e}", exc_info=True)
    return processed_artifacts

def _preprocess_ecg_stream(config, components, p_logger, streams, pid, output_dir):
    """Helper to preprocess the ECG stream and run HRV analysis."""
    p_logger.info("Preprocessing ECG stream...")
    processed_artifacts = {}
    try:
        ecg_df = streams['ecg_df']
        import numpy as np
        import os
        sfreq = 1 / np.mean(np.diff(ecg_df['time_sec'])) if len(ecg_df) > 1 else 0
        processed_artifacts['ecg_sfreq'] = sfreq
        processed_artifacts['ecg_duration_sec'] = len(ecg_df) / sfreq if sfreq > 0 else 0
        if sfreq > 0:
            ecg_results = components['ecg_preprocessor'].preprocess_ecg(
                ecg_signal=ecg_df['ecg_signal'].to_numpy(),
                ecg_sfreq=sfreq,
                participant_id=pid,
                output_dir=os.path.join(output_dir, "ECG_Preproc"))
            if ecg_results is not None:
                rpeaks_df = ecg_results[1] if isinstance(ecg_results, tuple) else ecg_results
                if isinstance(rpeaks_df, pd.DataFrame) and not rpeaks_df.empty:
                    processed_artifacts['rpeaks_df_out'] = rpeaks_df
                    p_logger.info("ECG R-peak detection completed.")
        else:
            p_logger.warning("Could not determine ECG sampling frequency. Skipping ECG/HRV processing.")
    except Exception as e:
        p_logger.error(f"An error occurred during ECG/HRV processing: {e}", exc_info=True)
    return processed_artifacts

def _preprocess_eda_stream(config, components, p_logger, streams, pid, output_dir):
    """Helper to preprocess the EDA stream."""
    p_logger.info("Preprocessing EDA stream...")
    processed_artifacts = {}
    try:
        eda_df = streams['eda_df']
        import numpy as np
        import os
        sfreq = 1 / np.mean(np.diff(eda_df['time_sec'])) if len(eda_df) > 1 else 0
        processed_artifacts['eda_sfreq'] = sfreq
        if sfreq > 0:
            eda_results = components['eda_preprocessor'].preprocess_eda(
                eda_signal_raw=eda_df['eda_signal'].to_numpy(),
                eda_sampling_rate=sfreq,
                participant_id=pid,
                output_dir=os.path.join(output_dir, "EDA_Preproc"))
            if eda_results is not None and isinstance(eda_results, tuple) and len(eda_results) == 2:
                phasic_df, tonic_df = eda_results
                if isinstance(phasic_df, pd.DataFrame) and not phasic_df.empty:
                    processed_artifacts['eda_processed_df'] = phasic_df
                if isinstance(tonic_df, pd.DataFrame) and not tonic_df.empty:
                    processed_artifacts['eda_tonic_df'] = tonic_df
                p_logger.info("EDA preprocessing completed and artifacts stored.")
        else:
            p_logger.warning("Could not determine EDA sampling frequency. Skipping EDA processing.")
    except Exception as e:
        p_logger.error(f"An error occurred during EDA processing: {e}", exc_info=True)
    return processed_artifacts

def _preprocess_fnirs_stream(config, components, p_logger, streams, pid, output_dir):
    """Helper to preprocess the fNIRS stream."""
    p_logger.info("Preprocessing fNIRS stream...")
    processed_artifacts = {}
    try:
        fnirs_config = {
            'beer_lambert_ppf': config.getfloat('FNIRS', 'beer_lambert_ppf'),
            'filter_band': (config.getfloat('FNIRS', 'filter_l_freq'), config.getfloat('FNIRS', 'filter_h_freq')),
            'short_channel_regression': config.getboolean('FNIRS', 'short_channel_regression', fallback=True),
            'motion_correction_method': config.get('FNIRS', 'motion_correction_method', fallback='none')
        }
        processed_hbo_hbr = components['fnirs_preprocessor'].process(
            fnirs_raw_intensity=streams['fnirs_cw_amplitude'],
            config=fnirs_config
        )
        if processed_hbo_hbr is not None:
            processed_artifacts['fnirs_processed_hbo_hbr'] = processed_hbo_hbr
            p_logger.info("fNIRS preprocessing (HbO/HbR conversion) completed.")
    except Exception as e:
        p_logger.error(f"An error occurred during fNIRS preprocessing: {e}", exc_info=True)
    return processed_artifacts 