import mne
from mne_icalabel import label_components # For automatic ICA component labeling
import logging
import os
import sys
import site
import pandas as pd
import numpy as np # Import numpy for NaN checks
import warnings
from typing import Optional, Tuple, List, Union, Dict, Any

class EEGPreprocessor:
    """
    Universal EEG preprocessing module for MNE Raw/RawArray objects.
    - Accepts a config dict with required and optional keys.
    - Fills in missing keys with class-level defaults.
    - Raises clear errors for missing required keys.
    - Usable in any project (no project-specific assumptions).

    Required config keys:
        - 'eeg_filter_band': tuple/list of (low, high) frequencies (e.g., (1.0, 40.0))
        - 'ica_n_components': int, float (0-1), or 'rank'
        - 'ica_random_state': int or None
        - 'ica_accept_labels': list of str (e.g., ['brain', 'other'])
        - 'ica_reject_threshold': float (0.0-1.0)
        - 'ica_method': str (e.g., 'fastica', 'infomax')
        - 'ica_extended': bool
    Optional config keys (with defaults):
        - 'eeg_reference': str (default: 'average')
        - 'eeg_reference_projection': bool (default: False)
        - 'filter_fir_design': str (default: 'firwin')
        - 'ica_max_iter': int or 'auto' (default: 'auto')
        - 'resample_sfreq': float or None (default: None)
        - 'ica_labeling_method': str (default: 'iclabel')
    """
    # Class-level defaults
    DEFAULT_EEG_REFERENCE = 'average'
    DEFAULT_EEG_REFERENCE_PROJECTION = False # Changed to False to align with ICLabel recommendations
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'
    DEFAULT_ICA_MAX_ITER: Union[str, int] = 'auto' # MNE default
    DEFAULT_RESAMPLE_SFREQ: Optional[float] = None # No resampling by default, or e.g., 250.0
    
    # Class-level constants for known string values to avoid "magic strings"
    _ICA_METHOD_ICLABEL = 'iclabel'
    _ICA_N_COMPONENTS_RANK = 'rank'
    _ICA_MAX_ITER_AUTO = 'auto'
    DEFAULT_ICA_LABELING_METHOD = _ICA_METHOD_ICLABEL # Method for mne_icalabel

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def _check_ica_backend(self) -> None:
        """
        Checks for available ICLabel backends, prioritizing PyTorch.
        This helps diagnose dependency issues and configures the environment for onnxruntime if it's the fallback.
        """
        # --- 1. Check for PyTorch (preferred backend) ---
        try:
            import torch
            self.logger.info(f"EEGPreprocessor - Found PyTorch backend for ICLabel, version {torch.__version__}. This will be used.")
            return # Success, PyTorch is available.
        except ImportError:
            self.logger.info("EEGPreprocessor - PyTorch backend not found. Checking for onnxruntime as a fallback.")

        # --- 2. Check for onnxruntime (fallback backend) ---
        try:
            # On Windows, we must add the DLL path for onnxruntime to work in parallel processes.
            if sys.platform == "win32" and hasattr(os, 'add_dll_directory'):
                paths = site.getsitepackages()
                if site.getusersitepackages() not in paths:
                    paths.append(site.getusersitepackages())
                for p in paths:
                    onnx_capi_path = os.path.join(p, 'onnxruntime', 'capi')
                    if os.path.isdir(onnx_capi_path):
                        os.add_dll_directory(onnx_capi_path)
                        self.logger.info(f"EEGPreprocessor - Added onnxruntime C-API path to DLL search: {onnx_capi_path}")
                        break # Found and added

            import onnxruntime
            self.logger.info(f"EEGPreprocessor - Found onnxruntime backend for ICLabel, version {onnxruntime.__version__}. This will be used.")
        except ImportError:
            self.logger.warning("EEGPreprocessor - CRITICAL: No ICLabel backend (PyTorch or onnxruntime) found. Automatic ICA labeling will fail.")
        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error while preparing onnxruntime as a fallback backend: {e}", exc_info=True)

    @staticmethod
    def default_config():
        """Return a default config dict for typical EEG preprocessing."""
        return {
            'eeg_filter_band': (1.0, 40.0),
            'ica_n_components': 30,
            'ica_random_state': 42,
            'ica_accept_labels': ['brain', 'Other'],
            'ica_reject_threshold': 0.8,
            'ica_method': 'infomax',
            'ica_extended': True,
            'eeg_reference': EEGPreprocessor.DEFAULT_EEG_REFERENCE,
            'eeg_reference_projection': EEGPreprocessor.DEFAULT_EEG_REFERENCE_PROJECTION,
            'filter_fir_design': EEGPreprocessor.DEFAULT_FILTER_FIR_DESIGN,
            'ica_max_iter': EEGPreprocessor.DEFAULT_ICA_MAX_ITER,
            'resample_sfreq': EEGPreprocessor.DEFAULT_RESAMPLE_SFREQ,
            'ica_labeling_method': EEGPreprocessor.DEFAULT_ICA_LABELING_METHOD
        }

    def _validate_and_resolve_config(self, eeg_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validates, converts, and sets defaults for the EEG configuration. Returns None on validation failure."""
        # Accept any dict-like config, fill in missing keys with defaults
        if not isinstance(eeg_config, dict):
            cfg = dict(eeg_config)
        else:
            cfg = eeg_config.copy()
        # Fill in missing optional keys with defaults
        cfg.setdefault('eeg_reference', self.DEFAULT_EEG_REFERENCE)
        cfg.setdefault('eeg_reference_projection', self.DEFAULT_EEG_REFERENCE_PROJECTION)
        cfg.setdefault('filter_fir_design', self.DEFAULT_FILTER_FIR_DESIGN)
        cfg.setdefault('ica_labeling_method', self.DEFAULT_ICA_LABELING_METHOD)
        cfg.setdefault('resample_sfreq', self.DEFAULT_RESAMPLE_SFREQ)
        cfg.setdefault('ica_max_iter', self.DEFAULT_ICA_MAX_ITER)
        # --- Validate required parameters ---
        required_keys = ['eeg_filter_band', 'ica_n_components', 'ica_random_state', 'ica_accept_labels', 'ica_reject_threshold', 'ica_method', 'ica_extended']
        for key in required_keys:
            if key not in cfg or cfg[key] is None:
                self.logger.error(f"EEGPreprocessor - Missing required config key: '{key}'.")
                return None
        # Validate eeg_filter_band
        band = cfg.get('eeg_filter_band')
        if not isinstance(band, (list, tuple)) or len(band) != 2 or not all(isinstance(x, (int, float, type(None))) for x in band) or (band[0] is None and band[1] is None):
            self.logger.error(f"Invalid 'eeg_filter_band': {band}. Must be a tuple of two numbers (e.g., [1.0, 40.0]).")
            return None
        # Validate ica_n_components
        n_comp = cfg.get('ica_n_components')
        if not (n_comp is None or (isinstance(n_comp, int) and n_comp > 0) or (isinstance(n_comp, float) and 0 < n_comp <= 1.0) or (isinstance(n_comp, str) and n_comp == self._ICA_N_COMPONENTS_RANK)):
            self.logger.error(f"Invalid 'ica_n_components': {n_comp}. Must be a positive int, a float (0-1), or 'rank'.")
            return None
        # Validate ica_random_state
        if not (cfg.get('ica_random_state') is None or isinstance(cfg.get('ica_random_state'), int)):
            self.logger.error(f"Invalid 'ica_random_state': {cfg.get('ica_random_state')}. Must be an integer.")
            return None
        # Validate ica_accept_labels
        accept_labels = cfg.get('ica_accept_labels', [])
        if not isinstance(accept_labels, list) or not all(isinstance(label, str) for label in accept_labels):
            self.logger.error(f"Invalid 'ica_accept_labels': {accept_labels}. Must be a list of strings.")
            return None
        # Validate ica_reject_threshold
        reject_thresh = cfg.get('ica_reject_threshold')
        if not isinstance(reject_thresh, (int, float)) or not (0.0 <= reject_thresh <= 1.0):
            self.logger.error(f"Invalid 'ica_reject_threshold': {reject_thresh}. Must be a number between 0.0 and 1.0.")
            return None
        # Validate ica_method
        if not isinstance(cfg.get('ica_method'), str):
            self.logger.error(f"Invalid 'ica_method': {cfg.get('ica_method')}. Must be a string.")
            return None
        # Validate ica_extended
        if not isinstance(cfg.get('ica_extended'), bool):
            self.logger.error(f"Invalid 'ica_extended': {cfg.get('ica_extended')}. Must be a boolean.")
            return None
        # Validate optional parameters (if present)
        # ica_max_iter
        max_iter = cfg.get('ica_max_iter')
        if isinstance(max_iter, str):
            if max_iter.isdigit():
                cfg['ica_max_iter'] = int(max_iter)
            elif max_iter.lower() == self._ICA_MAX_ITER_AUTO:
                cfg['ica_max_iter'] = self._ICA_MAX_ITER_AUTO
            else:
                cfg['ica_max_iter'] = self.DEFAULT_ICA_MAX_ITER
        elif not isinstance(max_iter, int) and max_iter is not self._ICA_MAX_ITER_AUTO:
            cfg['ica_max_iter'] = self.DEFAULT_ICA_MAX_ITER
        # resample_sfreq
        sfreq = cfg.get('resample_sfreq')
        if sfreq is not None and (not isinstance(sfreq, (int, float)) or sfreq <= 0):
            cfg['resample_sfreq'] = self.DEFAULT_RESAMPLE_SFREQ
        return cfg

    def process(self, raw_eeg: Union[mne.io.Raw, mne.io.RawArray], eeg_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocesses raw EEG data based on a configuration dictionary.

        This method performs resampling, filtering, referencing, and ICA-based artifact
        removal on a continuous EEG signal.

        Args:
            raw_eeg: The raw MNE data object to be processed.
            eeg_config: A dictionary containing all processing parameters. Expected keys include:
                - 'eeg_filter_band' (Tuple[Optional[float], Optional[float]]): Required. Low and high cut-off frequencies.
                - 'ica_n_components' (Union[int, float, str, None]): Required. Number of ICA components (e.g., 15, 0.99, 'rank').
                - 'ica_random_state' (Optional[int]): Required. Seed for ICA reproducibility.
                - 'ica_accept_labels' (List[str]): Required. List of ICLabel component types to keep (e.g., ['brain', 'other']).
                - 'ica_reject_threshold' (float): Required. Probability threshold to reject components (0.0 to 1.0).
                - 'ica_method' (str): Required. ICA algorithm (e.g., 'fastica', 'infomax').
                - 'ica_extended' (bool): Required. Whether to use extended Infomax.
                - 'resample_sfreq' (Optional[float]): Optional. Target sampling frequency for downsampling.
                - 'eeg_reference' (Optional[str]): Optional. Reference method (e.g., 'average').
                - 'eeg_reference_projection' (Optional[bool]): Optional. Whether to use projection for the reference.
                - 'filter_fir_design' (Optional[str]): Optional. FIR filter design (e.g., 'firwin').
                - 'ica_max_iter' (Optional[Union[int, str]]): Optional. Max iterations for ICA (e.g., 500, 'auto').
                - 'ica_labeling_method' (Optional[str]): Optional. Method for mne_icalabel (e.g., 'iclabel').

        Returns:
            A dictionary containing the processed MNE Raw object ('eeg_processed_raw') and
            an events array ('eeg_events_from_stim') if a stim channel was found,
            or None on critical error.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None

        # --- 1. Data Integrity & Configuration Validation ---
        # Step 1.1: Check for an available ICA backend before processing
        self._check_ica_backend()

        # Step 1.2: Validate, resolve, and set defaults for the configuration.
        final_config = self._validate_and_resolve_config(eeg_config)
        if final_config is None:
            self.logger.error("EEGPreprocessor - Configuration validation failed. Aborting processing.")
            return None

        # Step 1.3: Data integrity check. Ensure data is loaded and contains no NaNs/Infs.
        if isinstance(raw_eeg, pd.DataFrame):
            self.logger.error("EEGPreprocessor - Received a pandas DataFrame instead of an MNE Raw object. Please convert your EEG data to an MNE Raw or RawArray before preprocessing.")
            return None
        if hasattr(raw_eeg, '_data') and raw_eeg._data is None and raw_eeg.preload is False:
            try:
                raw_eeg.load_data(verbose=False)
            except Exception as e_load:
                self.logger.error(f"EEGPreprocessor - Failed to load data for integrity check: {e_load}", exc_info=True)
                return None
        raw_data = raw_eeg.get_data()
        if np.any(np.isnan(raw_data)) or np.any(np.isinf(raw_data)):
            self.logger.error("EEGPreprocessor - Raw EEG data contains NaN or Inf values. This can cause unpredictable errors in filtering or ICA. Aborting preprocessing for this participant.")
            return None

        # --- 2. Log Effective Configs and Start Processing ---
        self.logger.info(f"EEGPreprocessor - Starting EEG preprocessing with effective configs: "
                         f"FilterBand={final_config['eeg_filter_band']}, FIRDesign='{final_config['filter_fir_design']}', "
                         f"Reference='{final_config['eeg_reference']}' (Projection={final_config['eeg_reference_projection']}), "
                         f"ResampleSFreq={final_config['resample_sfreq']}, "
                         f"ICA Method='{final_config['ica_method']}' (Extended={final_config['ica_extended']}), "
                         f"ICA Components={final_config['ica_n_components']}, ICA MaxIter='{final_config['ica_max_iter']}', "
                         f"ICA LabelMethod='{final_config['ica_labeling_method']}', ICA AcceptLabels={final_config['ica_accept_labels']}, "
                         f"ICA RejectThreshold={final_config['ica_reject_threshold']}.")

        # --- PATCH: Ensure filter_fir_design is always a string ---
        if final_config.get('filter_fir_design', None) is None:
            self.logger.warning("EEGPreprocessor - 'filter_fir_design' was None or missing. Defaulting to 'firwin'.")
            final_config['filter_fir_design'] = self.DEFAULT_FILTER_FIR_DESIGN

        try:
            # --- 3. Preprocessing Steps ---
            # Use values from the 'final_config' dictionary from now on.
            if final_config['resample_sfreq'] is not None and final_config['resample_sfreq'] < raw_eeg.info['sfreq']:
                self.logger.info(f"EEGPreprocessor - Resampling EEG from {raw_eeg.info['sfreq']} Hz to {final_config['resample_sfreq']} Hz.")

                # To prevent the "Resampling of the stim channels caused event information to become unreliable"
                # warning, we temporarily change the channel type to 'misc' during resampling.
                # This is safe because we have already extracted the events.
                stim_ch_names = raw_eeg.copy().pick('stim').ch_names
                with warnings.catch_warnings():
                    # This warning is expected when changing stim channel types and is safe to ignore
                    # because we have already extracted event information.
                    warnings.filterwarnings('ignore', message='The unit for channel')
                    if stim_ch_names:
                        raw_eeg.set_channel_types({ch: 'misc' for ch in stim_ch_names})
                    raw_eeg.resample(sfreq=final_config['resample_sfreq'], n_jobs=-1, verbose=False)
                    if stim_ch_names:
                        raw_eeg.set_channel_types({ch: 'stim' for ch in stim_ch_names}) # Restore type

                self.logger.info(f"EEGPreprocessor - Resampling completed. New SFreq: {raw_eeg.info['sfreq']} Hz.")

            self.logger.info(f"EEGPreprocessor - Filtering EEG: {final_config['eeg_filter_band'][0]}-{final_config['eeg_filter_band'][1]} Hz.")
            raw_eeg.filter(l_freq=final_config['eeg_filter_band'][0], h_freq=final_config['eeg_filter_band'][1],
                           fir_design=final_config['filter_fir_design'], verbose=False)

            self.logger.info(f"EEGPreprocessor - Setting '{final_config['eeg_reference']}' reference (projection={final_config['eeg_reference_projection']}).")
            raw_eeg.set_eeg_reference(final_config['eeg_reference'], projection=final_config['eeg_reference_projection'], verbose=False)

            # --- 4. ICA Fitting ---
            # For optimal ICLabel performance, it's recommended to fit ICA on data
            # filtered between 1-100 Hz. We'll do this on a copy of the data to preserve
            # the original filtering for subsequent analysis steps.
            self.logger.info("EEGPreprocessor - Preparing a data copy for ICA fitting with a 1-100Hz filter as recommended by ICLabel.")
            raw_for_ica = raw_eeg.copy()
            raw_for_ica.filter(l_freq=1.0, h_freq=100.0, fir_design=final_config['filter_fir_design'], verbose=False)
            self.logger.info(f"EEGPreprocessor - Fitting ICA on data copy filtered at 1.0-100.0 Hz.")

            ica_instance = mne.preprocessing.ICA(
                n_components=final_config['ica_n_components'],
                method=str(final_config['ica_method']),
                fit_params=dict(extended=final_config['ica_extended']) if final_config['ica_method'] == 'infomax' else None,
                random_state=final_config['ica_random_state'],
                max_iter=final_config['ica_max_iter']) # type: ignore[arg-type]
            ica_instance.fit(raw_for_ica, verbose=False)

            # --- 5. ICA Labeling and Application ---
            self.logger.info("EEGPreprocessor - Attempting automatic ICA component labeling.")
            try:
                # Note: We label components using the ICA-specific filtered data for accuracy.
                with warnings.catch_warnings():
                    # ICLabel often issues a warning about data not being filtered 1-100Hz,
                    # even when we do it on a copy. We are following best practices, so we can
                    # safely filter this specific warning to keep logs clean.
                    warnings.filterwarnings("ignore", message="The provided Raw instance is not filtered between 1 and 100 Hz.", category=RuntimeWarning)
                    component_labels = label_components(raw_for_ica, ica_instance, method=final_config['ica_labeling_method'])
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]

                # Based on the IndexError, 'probabilities' is a 1D array where each element
                # corresponds to the probability of the winning label in the 'labels' list.
                # Therefore, we can directly use probabilities[idx] for the check.
                exclude_idx = [ # Indices of components to exclude
                    idx for idx, label in enumerate(labels)
                    if label not in final_config['ica_accept_labels'] and
                       probabilities[idx] > final_config['ica_reject_threshold']
                ]

                self.logger.info(f"EEGPreprocessor - Automatically identified {len(exclude_idx)} ICA components to exclude: {exclude_idx}")
                if exclude_idx: # Only apply if there are components to exclude
                    ica_instance.exclude = exclude_idx
                    # IMPORTANT: Apply the fitted ICA to the ORIGINAL raw data, not the ICA-specific filtered copy.
                    ica_instance.apply(raw_eeg, verbose=False)
                    self.logger.info("EEGPreprocessor - ICA applied to remove artifact components from original data.")
                else:
                    self.logger.info("EEGPreprocessor - No ICA components met criteria for automatic exclusion.")

            except ImportError as e_icalabel:
                self.logger.error(f"EEGPreprocessor - Automatic ICA labeling failed due to a missing dependency: {e_icalabel}. Please install a backend for ICLabel (e.g., 'pip install onnxruntime'). Since ICA is a critical step, preprocessing will be marked as failed.", exc_info=False)
                return None # Fail the entire preprocessing step if ICA backend is missing
            except Exception as e_icalabel:
                self.logger.warning(f"EEGPreprocessor - Automatic ICA labeling failed with an unexpected error: {e_icalabel}. ICA components not automatically excluded. Manual inspection might be needed.", exc_info=True)

            self.logger.info("EEGPreprocessor - EEG preprocessing completed.")
            return {'eeg_processed_raw': raw_eeg}

        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error during EEG preprocessing: {e}", exc_info=True)
            return None