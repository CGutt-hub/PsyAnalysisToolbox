import mne
from mne_icalabel import label_components # For automatic ICA component labeling
import logging
import pandas as pd
import numpy as np # Import numpy for NaN checks
from typing import Optional, Tuple, List, Union, Dict, Any

class EEGPreprocessor:
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

    def _validate_config(self, eeg_config: Dict[str, Any]) -> bool:
        """Validates the provided EEG configuration dictionary."""
        # --- Validate critical configurations for clarity and robustness ---
        eeg_filter_band_config = eeg_config.get('eeg_filter_band')
        if not isinstance(eeg_filter_band_config, (list, tuple)) or len(eeg_filter_band_config) != 2:
            self.logger.error(f"EEGPreprocessor - 'eeg_filter_band' must be a list or tuple of two elements. Got: {eeg_filter_band_config}. Skipping.")
            return False
        if not all(isinstance(x, (int, float)) or x is None for x in eeg_filter_band_config):
            self.logger.error(f"EEGPreprocessor - Elements of 'eeg_filter_band' must be numbers or None. Got: {eeg_filter_band_config}. Skipping.")
            return False
        if eeg_filter_band_config[0] is None and eeg_filter_band_config[1] is None:
            self.logger.error("EEGPreprocessor - 'eeg_filter_band' cannot have both elements as None. Skipping.")
            return False
        
        ica_n_components_config = eeg_config.get('ica_n_components')
        is_valid_ica_n_components = (
            ica_n_components_config is None or
            (isinstance(ica_n_components_config, int) and ica_n_components_config > 0) or
            (isinstance(ica_n_components_config, float) and 0 < ica_n_components_config <= 1.0) or
            (isinstance(ica_n_components_config, str) and ica_n_components_config == self._ICA_N_COMPONENTS_RANK)
        )
        if not is_valid_ica_n_components:
            self.logger.error(f"EEGPreprocessor - 'ica_n_components' must be None, a positive integer, a float between 0-1, or '{self._ICA_N_COMPONENTS_RANK}'. Got: {ica_n_components_config}. Skipping.")
            return False
        
        ica_random_state_config = eeg_config.get('ica_random_state')
        if not (ica_random_state_config is None or isinstance(ica_random_state_config, int)):
            self.logger.error(f"EEGPreprocessor - 'ica_random_state' must be None or an integer. Got: {ica_random_state_config}. Skipping.")
            return False
            
        ica_accept_labels_config = eeg_config.get('ica_accept_labels', [])
        if not isinstance(ica_accept_labels_config, list) or \
           not all(isinstance(label, str) for label in ica_accept_labels_config):
            self.logger.error("EEGPreprocessor - 'ica_accept_labels' must be a list of strings. Skipping.")
            return False
        
        ica_reject_threshold_config = eeg_config.get('ica_reject_threshold')
        if not isinstance(ica_reject_threshold_config, (int, float)) or not (0.0 <= ica_reject_threshold_config <= 1.0):
            self.logger.error(f"EEGPreprocessor - 'ica_reject_threshold' must be a number between 0.0 and 1.0. Got: {ica_reject_threshold_config}. Skipping.")
            return False
        
        return True # All validations passed

    def _resolve_final_configs(self, eeg_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves final configuration values, applying defaults for optional parameters."""
        resolved = eeg_config.copy() # Start with the provided config

        # Determine final config values, using defaults if not provided or invalid
        if not (isinstance(resolved.get('eeg_reference'), str) and resolved.get('eeg_reference', '').strip()):
            resolved['eeg_reference'] = self.DEFAULT_EEG_REFERENCE

        if not isinstance(resolved.get('eeg_reference_projection'), bool):
            resolved['eeg_reference_projection'] = self.DEFAULT_EEG_REFERENCE_PROJECTION

        if not (isinstance(resolved.get('filter_fir_design'), str) and resolved.get('filter_fir_design', '').strip()):
            resolved['filter_fir_design'] = self.DEFAULT_FILTER_FIR_DESIGN

        # Handle ica_max_iter
        ica_max_iter = resolved.get('ica_max_iter')
        if not (isinstance(ica_max_iter, int) or (isinstance(ica_max_iter, str) and ica_max_iter.strip().lower() == self._ICA_MAX_ITER_AUTO)):
            resolved['ica_max_iter'] = self.DEFAULT_ICA_MAX_ITER
        elif isinstance(ica_max_iter, str):
             resolved['ica_max_iter'] = ica_max_iter.strip().lower()

        if not (isinstance(resolved.get('ica_labeling_method'), str) and resolved.get('ica_labeling_method', '').strip()):
            resolved['ica_labeling_method'] = self.DEFAULT_ICA_LABELING_METHOD

        if not (isinstance(resolved.get('resample_sfreq'), (float, int)) and resolved.get('resample_sfreq', 0) > 0):
            resolved['resample_sfreq'] = self.DEFAULT_RESAMPLE_SFREQ
        else:
            resolved['resample_sfreq'] = float(resolved['resample_sfreq'])

        return resolved

    def process(self, raw_eeg: Union[mne.io.Raw, mne.io.RawArray], eeg_config: Dict[str, Any]) -> Optional[Union[mne.io.Raw, mne.io.RawArray]]:
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
            The preprocessed MNE Raw object, or None if a critical error occurs.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None

        ica_accept_labels_config = eeg_config.get('ica_accept_labels', [])
        ica_reject_threshold_config = eeg_config.get('ica_reject_threshold')
        ica_method_config = eeg_config.get('ica_method')
        ica_extended_config = eeg_config.get('ica_extended', False)
        # Optional
        eeg_reference_config = eeg_config.get('eeg_reference')
        eeg_reference_projection_config = eeg_config.get('eeg_reference_projection')
        filter_fir_design_config = eeg_config.get('filter_fir_design')
        ica_max_iter_config = eeg_config.get('ica_max_iter')
        resample_sfreq_config = eeg_config.get('resample_sfreq')
        ica_labeling_method_config = eeg_config.get('ica_labeling_method')

        # --- Pre-computation Data Integrity Check ---
        # Ensure data is loaded before checking for NaN/Inf values.
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

        # --- Validate critical configurations for clarity and robustness ---
        if not isinstance(eeg_filter_band_config, (list, tuple)) or len(eeg_filter_band_config) != 2:
            self.logger.error(f"EEGPreprocessor - 'eeg_filter_band' must be a list or tuple of two elements. Got: {eeg_filter_band_config}. Skipping.")
            return None
        if not all(isinstance(x, (int, float)) or x is None for x in eeg_filter_band_config):
            self.logger.error(f"EEGPreprocessor - Elements of 'eeg_filter_band' must be numbers or None. Got: {eeg_filter_band_config}. Skipping.")
            return None
        if eeg_filter_band_config[0] is None and eeg_filter_band_config[1] is None:
            self.logger.error("EEGPreprocessor - 'eeg_filter_band' cannot have both elements as None. Skipping.")
            return None
        
        is_valid_ica_n_components = (
            ica_n_components_config is None or
            (isinstance(ica_n_components_config, int) and ica_n_components_config > 0) or
            (isinstance(ica_n_components_config, float) and 0 < ica_n_components_config <= 1.0) or
            (isinstance(ica_n_components_config, str) and ica_n_components_config == 'rank')
        )
        if not is_valid_ica_n_components:
            self.logger.error(f"EEGPreprocessor - 'ica_n_components' must be None, a positive integer, a float between 0-1, or 'rank'. Got: {ica_n_components_config}. Skipping.")
            return None
        
        if not (ica_random_state_config is None or isinstance(ica_random_state_config, int)):
            self.logger.error(f"EEGPreprocessor - 'ica_random_state' must be None or an integer. Got: {ica_random_state_config}. Skipping.")
            return None
            
        # ica_random_state_config can be None (MNE default).
        if not isinstance(ica_accept_labels_config, list) or \
           not all(isinstance(label, str) for label in ica_accept_labels_config):
            self.logger.error("EEGPreprocessor - 'ica_accept_labels_config' must be a list of strings. Skipping.")
            return None
        
        if not isinstance(ica_reject_threshold_config, (int, float)) or not (0.0 <= ica_reject_threshold_config <= 1.0):
            self.logger.error(f"EEGPreprocessor - 'ica_reject_threshold' must be a number between 0.0 and 1.0. Got: {ica_reject_threshold_config}. Skipping.")
            return None

        # Determine final config values, using defaults if not provided or invalid
        final_eeg_ref = self.DEFAULT_EEG_REFERENCE
        if eeg_reference_config is not None:
            if isinstance(eeg_reference_config, str) and eeg_reference_config.strip():
                final_eeg_ref = eeg_reference_config.strip()
            else:
                self.logger.warning(f"EEGPreprocessor: Invalid 'eeg_reference_config' ('{eeg_reference_config}'). Using default: '{self.DEFAULT_EEG_REFERENCE}'.")

        final_eeg_ref_proj = self.DEFAULT_EEG_REFERENCE_PROJECTION
        if eeg_reference_projection_config is not None:
            if isinstance(eeg_reference_projection_config, bool):
                final_eeg_ref_proj = eeg_reference_projection_config
            else:
                self.logger.warning(f"EEGPreprocessor: Invalid 'eeg_reference_projection_config' ('{eeg_reference_projection_config}'). Using default: '{self.DEFAULT_EEG_REFERENCE_PROJECTION}'.")

        final_fir_design = self.DEFAULT_FILTER_FIR_DESIGN
        if filter_fir_design_config is not None:
            if isinstance(filter_fir_design_config, str) and filter_fir_design_config.strip():
                final_fir_design = filter_fir_design_config.strip()
            else:
                self.logger.warning(f"EEGPreprocessor: Invalid 'filter_fir_design_config' ('{filter_fir_design_config}'). Using default: '{self.DEFAULT_FILTER_FIR_DESIGN}'.")

        # Determine final_ica_max_iter with more explicit type handling
        final_ica_max_iter: Union[str, int] # Declare type
        if ica_max_iter_config is None:
            final_ica_max_iter = self.DEFAULT_ICA_MAX_ITER
        elif isinstance(ica_max_iter_config, int):
            final_ica_max_iter = ica_max_iter_config
        elif isinstance(ica_max_iter_config, str) and ica_max_iter_config.strip().lower() == 'auto':
            final_ica_max_iter = ica_max_iter_config.strip().lower()
        else:
            self.logger.warning(f"EEGPreprocessor: Invalid 'ica_max_iter_config' ('{ica_max_iter_config}'). Using default: '{self.DEFAULT_ICA_MAX_ITER}'.")
            final_ica_max_iter = self.DEFAULT_ICA_MAX_ITER

        final_ica_label_method = self.DEFAULT_ICA_LABELING_METHOD
        if ica_labeling_method_config is not None:
            if isinstance(ica_labeling_method_config, str) and ica_labeling_method_config.strip():
                final_ica_label_method = ica_labeling_method_config.strip()
            else:
                self.logger.warning(f"EEGPreprocessor: Invalid 'ica_labeling_method_config' ('{ica_labeling_method_config}'). Using default: '{self.DEFAULT_ICA_LABELING_METHOD}'.")

        final_resample_sfreq = self.DEFAULT_RESAMPLE_SFREQ
        if resample_sfreq_config is not None:
            if isinstance(resample_sfreq_config, (float, int)) and resample_sfreq_config > 0:
                final_resample_sfreq = float(resample_sfreq_config)
            else:
                self.logger.warning(f"EEGPreprocessor: Invalid 'resample_sfreq_config' ('{resample_sfreq_config}'). Using default: '{self.DEFAULT_RESAMPLE_SFREQ}'.")


        self.logger.info(f"EEGPreprocessor - Starting EEG preprocessing with effective configs: "
                         f"FilterBand={eeg_filter_band_config}, FIRDesign='{final_fir_design}', "
                         f"Reference='{final_eeg_ref}' (Projection={final_eeg_ref_proj}), "
                         f"ResampleSFreq={final_resample_sfreq}, "
                         f"ICA Method='{ica_method_config}' (Extended={ica_extended_config}), "
                         f"ICA Components={ica_n_components_config}, ICA MaxIter='{final_ica_max_iter}', "
                         f"ICA LabelMethod='{final_ica_label_method}', ICA AcceptLabels={ica_accept_labels_config}, "
                         f"ICA RejectThreshold={ica_reject_threshold_config}.")

        try:
            # Resample if configured
            if final_resample_sfreq is not None and final_resample_sfreq < raw_eeg.info['sfreq']:
                self.logger.info(f"EEGPreprocessor - Resampling EEG from {raw_eeg.info['sfreq']} Hz to {final_resample_sfreq} Hz.")
                raw_eeg.resample(sfreq=final_resample_sfreq, verbose=False)
                self.logger.info(f"EEGPreprocessor - Resampling completed. New SFreq: {raw_eeg.info['sfreq']} Hz.")

            self.logger.info(f"EEGPreprocessor - Filtering EEG: {eeg_filter_band_config[0]}-{eeg_filter_band_config[1]} Hz.")
            raw_eeg.filter(l_freq=eeg_filter_band_config[0], h_freq=eeg_filter_band_config[1], 
                           fir_design=final_fir_design, verbose=False)

            self.logger.info(f"EEGPreprocessor - Setting '{final_eeg_ref}' reference (projection={final_eeg_ref_proj}).")
            raw_eeg.set_eeg_reference(final_eeg_ref, projection=final_eeg_ref_proj, verbose=False)

            # --- ICA Fitting ---
            # For optimal ICLabel performance, it's recommended to fit ICA on data
            # filtered between 1-100 Hz. We'll do this on a copy of the data to preserve
            # the original filtering for subsequent analysis steps.
            self.logger.info("EEGPreprocessor - Preparing a data copy for ICA fitting with a 1-100Hz filter as recommended by ICLabel.")
            raw_for_ica = raw_eeg.copy()
            raw_for_ica.filter(l_freq=1.0, h_freq=100.0, fir_design=final_fir_design, verbose=False)
            self.logger.info(f"EEGPreprocessor - Fitting ICA on data copy filtered at 1.0-100.0 Hz.")

            # Instantiate ICA using final_ica_max_iter directly.
            ica_instance = mne.preprocessing.ICA(
                n_components=ica_n_components_config,
                method=str(ica_method_config),
                fit_params=dict(extended=ica_extended_config) if ica_method_config == 'infomax' else None,
                random_state=ica_random_state_config,
                max_iter=final_ica_max_iter) # type: ignore[arg-type]
            ica_instance.fit(raw_for_ica, verbose=False)

            # Automatic artifact labeling (optional, requires mne_icalabel)
            self.logger.info("EEGPreprocessor - Attempting automatic ICA component labeling.")
            try:
                # Note: We label components using the ICA-specific filtered data for accuracy.
                component_labels = label_components(raw_for_ica, ica_instance, method=final_ica_label_method)
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]

                exclude_idx = [ # Indices of components to exclude
                    idx for idx, label in enumerate(labels)
                    if label not in ica_accept_labels_config and # label is already a string
                       probabilities[idx, list(component_labels['classes']).index(label)] > ica_reject_threshold_config
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
            return raw_eeg # Return the processed MNE Raw object

        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error during EEG preprocessing: {e}", exc_info=True)
            return None