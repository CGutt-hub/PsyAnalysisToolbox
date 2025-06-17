import mne
from mne_icalabel import label_components # For automatic ICA component labeling
from typing import Optional, Tuple, List, Union

class EEGPreprocessor:
    # Class-level defaults
    DEFAULT_EEG_REFERENCE = 'average'
    DEFAULT_EEG_REFERENCE_PROJECTION = True
    DEFAULT_FILTER_FIR_DESIGN = 'firwin'
    DEFAULT_ICA_MAX_ITER: Union[str, int] = 'auto' # MNE default
    DEFAULT_ICA_LABELING_METHOD = 'iclabel' # Method for mne_icalabel

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def process(self,
                  raw_eeg: mne.io.Raw,
                  # Critical configs (must be provided by orchestrator)
                  eeg_filter_band_config: Tuple[Optional[float], Optional[float]],
                  ica_n_components_config: Optional[Union[int, float, str]], # Can be None, int, float, or 'rank'
                  ica_random_state_config: Optional[int], # Can be None for MNE default
                  ica_accept_labels_config: List[str], # List of labels to accept
                  ica_reject_threshold_config: float, # Threshold for rejection
                  # Optional configs with internal defaults
                  eeg_reference_config: Optional[str] = None,
                  eeg_reference_projection_config: Optional[bool] = None,
                  filter_fir_design_config: Optional[str] = None,
                  ica_max_iter_config: Optional[Union[int, str]] = None,
                  ica_labeling_method_config: Optional[str] = None
                  ) -> Optional[mne.io.Raw]:
        """
        Preprocesses raw EEG data.
        Args:
            raw_eeg (mne.io.Raw): Raw EEG data.
            eeg_filter_band_config (tuple): Low and high cut-off frequencies for filtering (e.g., (0.5, 40.0)).
            ica_n_components_config (int or float or None): Number of ICA components.
            ica_random_state_config (int or None): Random state for ICA.
            ica_accept_labels_config (list): List of ICA component labels to keep (e.g., ['brain', 'other']).
            ica_reject_threshold_config (float): Probability threshold to reject components not in accept_labels.
            eeg_reference_config (Optional[str]): EEG reference method. Defaults to DEFAULT_EEG_REFERENCE.
            eeg_reference_projection_config (Optional[bool]): Whether to use projection for reference. Defaults to DEFAULT_EEG_REFERENCE_PROJECTION.
            filter_fir_design_config (Optional[str]): FIR filter design. Defaults to DEFAULT_FILTER_FIR_DESIGN.
            ica_max_iter_config (Optional[Union[int, str]]): Max iterations for ICA. Defaults to DEFAULT_ICA_MAX_ITER.
            ica_labeling_method_config (Optional[str]): Method for mne_icalabel. Defaults to DEFAULT_ICA_LABELING_METHOD.
        Returns:
            mne.io.Raw: Preprocessed EEG data, or None if input is None or error.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None

        # Validate critical configurations
        if not (isinstance(eeg_filter_band_config, (list, tuple)) and \
                len(eeg_filter_band_config) == 2 and \
                all(isinstance(x, (int, float)) or x is None for x in eeg_filter_band_config) and \
                not (eeg_filter_band_config[0] is None and eeg_filter_band_config[1] is None)): # Both cannot be None
            self.logger.error("EEGPreprocessor - 'eeg_filter_band_config' must be a list/tuple of two (int, float, or None), with at least one non-None. Skipping.")
            return None
        
        # Validate ica_n_components_config
        if not (ica_n_components_config is None or \
                isinstance(ica_n_components_config, int) or \
                (isinstance(ica_n_components_config, float) and 0 < ica_n_components_config <= 1) or \
                (isinstance(ica_n_components_config, str) and ica_n_components_config == 'rank')):
            self.logger.error("EEGPreprocessor - 'ica_n_components_config' must be None, int, float (0-1 exclusive of 0), or 'rank'. Skipping.")
            return None
        
        if not (ica_random_state_config is None or isinstance(ica_random_state_config, int)):
            self.logger.error("EEGPreprocessor - 'ica_random_state_config' must be None or an integer. Skipping.")
            return None
            
        # ica_random_state_config can be None (MNE default).
        if not isinstance(ica_accept_labels_config, list) or \
           not all(isinstance(label, str) for label in ica_accept_labels_config):
            self.logger.error("EEGPreprocessor - 'ica_accept_labels_config' must be a list of strings. Skipping.")
            return None
        if not (isinstance(ica_reject_threshold_config, (int, float)) and \
                0.0 <= ica_reject_threshold_config <= 1.0):
            self.logger.error("EEGPreprocessor - 'ica_reject_threshold_config' must be a number between 0.0 and 1.0 (inclusive). Skipping.")
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


        self.logger.info(f"EEGPreprocessor - Starting EEG preprocessing with effective configs: "
                         f"FilterBand={eeg_filter_band_config}, FIRDesign='{final_fir_design}', "
                         f"Reference='{final_eeg_ref}' (Projection={final_eeg_ref_proj}), "
                         f"ICA Components={ica_n_components_config}, ICA MaxIter='{final_ica_max_iter}', "
                         f"ICA LabelMethod='{final_ica_label_method}', ICA AcceptLabels={ica_accept_labels_config}, "
                         f"ICA RejectThreshold={ica_reject_threshold_config}.")

        try:
            # Ensure data is loaded if it's not already
            if hasattr(raw_eeg, '_data') and raw_eeg._data is None and raw_eeg.preload is False:
                 raw_eeg.load_data(verbose=False)

            self.logger.info(f"EEGPreprocessor - Filtering EEG: {eeg_filter_band_config[0]}-{eeg_filter_band_config[1]} Hz.")
            raw_eeg.filter(l_freq=eeg_filter_band_config[0], h_freq=eeg_filter_band_config[1], 
                           fir_design=final_fir_design, verbose=False)

            self.logger.info(f"EEGPreprocessor - Setting '{final_eeg_ref}' reference (projection={final_eeg_ref_proj}).")
            raw_eeg.set_eeg_reference(final_eeg_ref, projection=final_eeg_ref_proj, verbose=False)

            self.logger.info(f"EEGPreprocessor - Fitting ICA with {ica_n_components_config} components, max_iter='{final_ica_max_iter}'.")

            # Instantiate ICA using final_ica_max_iter directly.
            # MNE API for max_iter accepts int or 'auto' (str).
            ica_instance = mne.preprocessing.ICA(n_components=ica_n_components_config,
                                        random_state=ica_random_state_config,
                                        max_iter=final_ica_max_iter) # type: ignore[arg-type]

            ica_instance.fit(raw_eeg, verbose=False)


            # Automatic artifact labeling (optional, requires mne_icalabel)
            self.logger.info("EEGPreprocessor - Attempting automatic ICA component labeling.")
            try:
                component_labels = label_components(raw_eeg, ica_instance, method=final_ica_label_method)
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]
                
                # ica_accept_labels_config is already validated to be a list of strings.
                
                exclude_idx = [ # Indices of components to exclude
                    idx for idx, label in enumerate(labels)
                    if label not in ica_accept_labels_config and # label is already a string
                       probabilities[idx, list(component_labels['classes']).index(label)] > ica_reject_threshold_config
                ]
                
                self.logger.info(f"EEGPreprocessor - Automatically identified {len(exclude_idx)} ICA components to exclude: {exclude_idx}")
                if exclude_idx: # Only apply if there are components to exclude
                    ica_instance.exclude = exclude_idx
                    ica_instance.apply(raw_eeg, verbose=False) # Apply ICA to the raw data
                    self.logger.info("EEGPreprocessor - ICA applied to remove artifact components.")
                else:
                    self.logger.info("EEGPreprocessor - No ICA components met criteria for automatic exclusion.")

            except Exception as e_icalabel:
                self.logger.warning(f"EEGPreprocessor - Automatic ICA labeling failed: {e_icalabel}. ICA components not automatically excluded. Manual inspection might be needed.", exc_info=True)

            self.logger.info("EEGPreprocessor - EEG preprocessing completed.")
            return raw_eeg
        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error during EEG preprocessing: {e}", exc_info=True)
            return None