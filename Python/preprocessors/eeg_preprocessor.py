"""
EEG Preprocessor Module
----------------------
Universal EEG preprocessing for MNE Raw/RawArray objects.
Handles filtering, referencing, ICA (Infomax+ICLabel), and debug mode for fast testing.
Config-driven, robust, and maintainable.
"""
import mne
from mne_icalabel import label_components
import logging
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple, List, Union, Dict, Any
from PsyAnalysisToolbox.Python.utils.logging_utils import log_progress_bar

class EEGPreprocessor:
    """
    Universal EEG preprocessing module for MNE Raw/RawArray objects.
    - Accepts a config dict with required and optional keys.
    - Fills in missing keys with class-level defaults.
    - Raises clear errors for missing required keys.
    - Usable in any project (no project-specific assumptions).
    """
    # Class-level defaults
    DEFAULT_EEG_REFERENCE = 'average'
    DEFAULT_EEG_REFERENCE_PROJECTION = False
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'
    DEFAULT_ICA_MAX_ITER: Union[str, int] = 'auto'
    DEFAULT_RESAMPLE_SFREQ: Optional[float] = None
    _ICA_METHOD_ICLABEL = 'iclabel'
    _ICA_N_COMPONENTS_RANK = 'rank'
    _ICA_MAX_ITER_AUTO = 'auto'
    DEFAULT_ICA_LABELING_METHOD = _ICA_METHOD_ICLABEL

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def process(self, raw_eeg: Union[mne.io.Raw, mne.io.RawArray], eeg_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point for EEG preprocessing.
        Applies filtering, referencing, ICA (Infomax+ICLabel), and debug mode if enabled.
        Returns a dictionary with the processed raw EEG.
        """
        # Debug mode for fast pipeline testing
        if eeg_config.get('debug_mode', False):
            self.logger.warning("EEGPreprocessor: DEBUG MODE ACTIVE - Cropping to 30s, using first 8 channels, ica_n_components=5.")
            if hasattr(raw_eeg, 'crop'):
                raw_eeg.crop(tmin=0, tmax=min(30, raw_eeg.times[-1]))
            if len(raw_eeg.ch_names) > 8:
                raw_eeg.pick_channels(raw_eeg.ch_names[:8])
            eeg_config['ica_n_components'] = 5
            data = np.asarray(raw_eeg.get_data())
            self.logger.info(f"EEGPreprocessor: After debug crop, data shape: {data.shape} (channels, samples)")
            # Fallback crop if still too large
            shape = data.shape
            if shape[0] > 8 or shape[1] > 3000:
                self.logger.warning(f"EEGPreprocessor: Fallback crop - data still too large after debug crop (channels={shape[0]}, samples={shape[1]}). Cropping to 8 channels, 10 seconds.")
                if hasattr(raw_eeg, 'crop'):
                    raw_eeg.crop(tmin=0, tmax=min(10, raw_eeg.times[-1]))
                if len(raw_eeg.ch_names) > 8:
                    raw_eeg.pick_channels(raw_eeg.ch_names[:8])
                data = np.asarray(raw_eeg.get_data())
                self.logger.info(f"EEGPreprocessor: After fallback crop, data shape: {data.shape} (channels, samples)")

        # Data integrity check
        if isinstance(raw_eeg, pd.DataFrame):
            if raw_eeg.isnull().values.any():
                self.logger.error("EEGPreprocessor: NaNs detected in input DataFrame.")
                return None
        elif isinstance(raw_eeg, (mne.io.Raw, mne.io.RawArray)):
            if np.isnan(raw_eeg.get_data()).any():
                self.logger.error("EEGPreprocessor: NaNs detected in input Raw object.")
                return None

        steps = 5
        update, close = log_progress_bar(self.logger, steps, desc="EEG", per_process=True)
        # Filtering
        update(); self.logger.info("EEGPreprocessor: Filtering")
        filter_band = eeg_config.get('eeg_filter_band', (1.0, 40.0))
        raw_eeg.filter(l_freq=filter_band[0], h_freq=filter_band[1], fir_design=eeg_config.get('filter_fir_design', self.DEFAULT_FILTER_FIR_DESIGN), verbose=False)
        self.logger.info(f"EEGPreprocessor: Filtered EEG {filter_band[0]}-{filter_band[1]} Hz.")
        # Referencing
        update(); self.logger.info("EEGPreprocessor: Referencing")
        reference = eeg_config.get('eeg_reference', self.DEFAULT_EEG_REFERENCE)
        projection = eeg_config.get('eeg_reference_projection', self.DEFAULT_EEG_REFERENCE_PROJECTION)
        raw_eeg.set_eeg_reference(reference, projection=projection, verbose=False)
        self.logger.info(f"EEGPreprocessor: Set reference to '{reference}' (projection={projection}).")
        # ICA (Infomax+ICLabel)
        if eeg_config.get('ica_method', 'infomax').lower() == 'infomax' and eeg_config.get('ica_labeling_method', '').lower() == 'iclabel':
            data = np.asarray(raw_eeg.get_data())
            update(); self.logger.info("EEGPreprocessor: Running ICA")
            self.logger.info(f"EEGPreprocessor: ICA input shape: {data.shape}, n_components: {eeg_config.get('ica_n_components')}")
            import time
            ica_start = time.time()
            self.logger.info("EEGPreprocessor: Starting ICA fit (this may take a few seconds for small data, longer for large data)...")
            # Simulate a loading bar with periodic logging if ICA takes long
            import threading
            ica_progress = {'running': True}
            def ica_loading_bar():
                waited = 0
                while ica_progress['running']:
                    time.sleep(5)
                    waited += 5
                    if ica_progress['running']:
                        self.logger.info(f"EEGPreprocessor: ICA is still running... ({waited} seconds elapsed)")
            progress_thread = threading.Thread(target=ica_loading_bar)
            progress_thread.start()
            ica = mne.preprocessing.ICA(
                n_components=eeg_config.get('ica_n_components', 30),
                method='infomax',
                fit_params=dict(extended=True),
                random_state=eeg_config.get('ica_random_state', 42),
                max_iter=eeg_config.get('ica_max_iter', self.DEFAULT_ICA_MAX_ITER)
            )
            ica.fit(raw_eeg, verbose=False)
            ica_progress['running'] = False
            progress_thread.join()
            ica_elapsed = time.time() - ica_start
            self.logger.info(f"EEGPreprocessor: ICA fit completed in {ica_elapsed:.2f} seconds.")
            iclabel_start = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The provided Raw instance is not filtered between 1 and 100 Hz.", category=RuntimeWarning)
                    component_labels = label_components(raw_eeg, ica, method=eeg_config.get('ica_labeling_method', self.DEFAULT_ICA_LABELING_METHOD))
                iclabel_elapsed = time.time() - iclabel_start
                self.logger.info(f"EEGPreprocessor: ICLabel completed in {iclabel_elapsed:.2f} seconds.")
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]
                exclude_idx = [
                    idx for idx, label in enumerate(labels)
                    if label not in eeg_config.get('ica_accept_labels', ['brain', 'Other']) and probabilities[idx] > eeg_config.get('ica_reject_threshold', 0.8)
                ]
                self.logger.info(f"EEGPreprocessor: ICLabel identified {len(exclude_idx)} components to exclude: {exclude_idx}")
                if exclude_idx:
                    ica.exclude = exclude_idx
                    ica.apply(raw_eeg, verbose=False)
                    self.logger.info("EEGPreprocessor: ICLabel artifact components removed from original data.")
                else:
                    self.logger.info("EEGPreprocessor: No ICA components met criteria for automatic exclusion (ICLabel).")
            except ImportError as e_icalabel:
                self.logger.error(f"EEGPreprocessor: ICLabel dependency missing: {e_icalabel}")
                return None
            except Exception as e_icalabel:
                self.logger.warning(f"EEGPreprocessor: ICLabel failed: {e_icalabel}", exc_info=True)
        else:
            self.logger.info("EEGPreprocessor: No automatic artifact rejection method configured for this ICA method. No components will be excluded.")

        update(); self.logger.info("EEGPreprocessor: Done")
        close()
        self.logger.info("EEGPreprocessor: EEG preprocessing completed.")
        return {'eeg_processed_raw': raw_eeg}