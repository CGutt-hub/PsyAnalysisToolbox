import pandas as pd
import mne
from mne_nirs.statistics import run_glm, RegressionResults # Use RegressionResults
from typing import Dict, List, Optional, Union # For type hinting

class GLMAnalyzer:
    # Class-level default parameters for GLM
    DEFAULT_GLM_NOISE_MODEL = "ar1"

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("GLMAnalyzer initialized.")

    def run_first_level_glm(self,
                          data_for_glm: mne.io.BaseRaw,
                          design_matrix_prepared: pd.DataFrame,
                          participant_id: str, # Participant ID is useful for logging context
                          contrasts_config: Dict[str, Dict[str, float]],
                          # Optional: fNIRS-specific ROI analysis parameters
                          rois_config: Optional[Dict[str, List[str]]] = None,
                          activation_p_threshold: Optional[float] = None,
                          roi_trigger_contrast_name: Optional[str] = None,
                          noise_model: str = DEFAULT_GLM_NOISE_MODEL # Uses class default if not specified
                          ) -> Dict[str, Union[Dict[str, RegressionResults], Dict[str, pd.DataFrame], List[str], None]]:
        """
        Runs a first-level GLM on prepared data using a prepared design matrix and computes specified contrasts.
        Optionally performs ROI-based analysis if ROI configurations are provided (typically for fNIRS).

        Args:
            data_for_glm (mne.io.BaseRaw): Data ready for GLM (e.g., mne.io.RawArray of concatenated evoked responses).
            design_matrix_prepared (pd.DataFrame): The pre-computed design matrix.
            participant_id (str): The ID of the participant.
            contrasts_config (Dict[str, Dict[str, float]]): Dictionary defining contrasts based on columns in design_matrix_prepared.
                                                            Format: {'ContrastName': {'condition1': weight1, 'condition2': weight2, ...}}
            rois_config (Optional[Dict[str, List[str]]]): Dictionary defining ROIs and their channels (e.g., for fNIRS).
                                                         Format: {'ROIName': ['channel1', 'channel2', ...]}
            activation_p_threshold (Optional[float]): P-value threshold for identifying active channels/ROIs (e.g., for fNIRS).
                                                      Must be between 0.0 and 1.0 if provided.
            roi_trigger_contrast_name (Optional[str]): The specific contrast name that should trigger ROI analysis. If None, ROI analysis is not triggered by p-value.
            noise_model (str): The noise model to use for the GLM. Defaults to GLMAnalyzer.DEFAULT_GLM_NOISE_MODEL.

        Returns:
            Dict[str, Union[Dict[str, RegressionResults], Dict[str, pd.DataFrame], List[str], None]]:
                A dictionary containing:
                'glm_estimates': Dict mapping channel names to GLMEstimate objects, or None on failure.
                'contrast_results': Dict mapping contrast names to DataFrames of results, or empty dict if no contrasts computed.
                'active_rois': List of ROI names identified as active, or empty list if no ROI analysis or no active ROIs.
                Returns a dict with None/empty values on critical failure before any results can be generated.
        """
        # Initialize return structure
        initial_return_state: Dict[str, Union[Dict[str, RegressionResults], Dict[str, pd.DataFrame], List[str], None]] = {
            'glm_estimates': None,
            'contrast_results': {},
            'active_rois': []
        }

        if data_for_glm is None or not isinstance(data_for_glm, mne.io.BaseRaw):
            self.logger.warning("GLMAnalyzer: data_for_glm is missing or not a Raw object. Skipping GLM.")
            return initial_return_state
        if design_matrix_prepared is None or not isinstance(design_matrix_prepared, pd.DataFrame) or design_matrix_prepared.empty:
            self.logger.warning("GLMAnalyzer: design_matrix_prepared is missing, not a DataFrame, or empty. Skipping GLM.")
            return initial_return_state
        
        self.logger.info(f"GLMAnalyzer: Running GLM for P:{participant_id}")
        
        if not isinstance(contrasts_config, dict) or not contrasts_config:
            self.logger.error("GLMAnalyzer: contrasts_config must be a non-empty dictionary.")
            return initial_return_state
        # Validate contrasts_config structure (basic check)
        if not all(isinstance(name, str) and isinstance(weights, dict) and all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in weights.items()) for name, weights in contrasts_config.items()):
             self.logger.error("GLMAnalyzer: contrasts_config has invalid structure. Expected Dict[str, Dict[str, float]]. Skipping.")
             return initial_return_state

        # Validate optional ROI configs if provided
        if rois_config is not None:
            if not isinstance(rois_config, dict) or not all(isinstance(roi_name, str) and isinstance(channels, list) and all(isinstance(ch, str) for ch in channels) for roi_name, channels in rois_config.items()):
                self.logger.warning("GLMAnalyzer: rois_config has invalid structure. Expected Dict[str, List[str]]. Skipping ROI analysis.")
                return initial_return_state
            for roi_name, channels in rois_config.items():
                invalid_channels = [channel for channel in channels if channel not in data_for_glm.ch_names]
                if invalid_channels:
                    self.logger.error(f"ROI {roi_name} contains invalid channels: {invalid_channels}")
                    return initial_return_state

                rois_config = None # Disable ROI analysis
            
            if activation_p_threshold is not None:
                if not (isinstance(activation_p_threshold, (int, float)) and 0.0 <= activation_p_threshold <= 1.0):
                    self.logger.warning(f"GLMAnalyzer: activation_p_threshold must be a float between 0.0 and 1.0. Got {activation_p_threshold}. Skipping ROI analysis.")
                    rois_config = None # Disable ROI analysis
            elif rois_config: # rois_config is valid, but no p_threshold
                 self.logger.info("GLMAnalyzer: rois_config provided, but activation_p_threshold is None. Skipping p-value based ROI analysis.")
                 rois_config = None # Disable p-value based ROI analysis

            if roi_trigger_contrast_name is not None and not isinstance(roi_trigger_contrast_name, str):
                 self.logger.warning(f"GLMAnalyzer: roi_trigger_contrast_name must be a string or None. Got {roi_trigger_contrast_name}. Skipping ROI analysis.")
                 rois_config = None # Disable ROI analysis

        # Validate noise_model
        if not isinstance(noise_model, str) or not noise_model.strip():
            self.logger.warning(f"GLMAnalyzer: Invalid noise_model ('{noise_model}'). Using default '{self.DEFAULT_GLM_NOISE_MODEL}'.")
            noise_model = self.DEFAULT_GLM_NOISE_MODEL

        # Validate that conditions in contrasts exist in the prepared design matrix columns
        design_matrix_columns = set(design_matrix_prepared.columns)
        for contrast_name, contrast_weights in contrasts_config.items():
            for cond_in_contrast in contrast_weights.keys():
                if cond_in_contrast not in design_matrix_columns:
                    self.logger.warning(f"GLMAnalyzer: Condition '{cond_in_contrast}' in contrast '{contrast_name}' not found in design_matrix_prepared columns. This contrast may fail or produce unexpected results.")

        try:
            # Filter the provided design_matrix_prepared to only include conditions relevant to the
            # specified contrasts, plus standard nuisance regressors (intercept, drifts).
            all_contrast_conditions = set()
            for contrast_name, contrast_weights_dict in contrasts_config.items():
                if isinstance(contrast_weights_dict, dict): # Ensure it's a dict before .keys()
                    all_contrast_conditions.update(contrast_weights_dict.keys())
                else:
                    self.logger.warning(f"GLMAnalyzer: Weights for contrast '{contrast_name}' are not a dictionary. Skipping conditions from this contrast for filtering.")
                    continue
            
            design_matrix_cols_to_keep = [
                col for col in design_matrix_prepared.columns
                if col in all_contrast_conditions or col.lower() == 'intercept' or 'drift' in col.lower() or 'constant' in col.lower()
            ]
            design_matrix_for_glm = design_matrix_prepared[design_matrix_cols_to_keep]

            if design_matrix_for_glm.empty or not any(cond in design_matrix_for_glm.columns for cond in all_contrast_conditions):
                 self.logger.warning(f"GLMAnalyzer: Design matrix empty or missing contrast conditions after filtering. Original design cols: {list(design_matrix_prepared.columns)}, Filtered for GLM: {list(design_matrix_for_glm.columns)}. Skipping GLM.")
                 return initial_return_state

            self.logger.info(f"GLMAnalyzer: Using design matrix with columns for GLM: {list(design_matrix_for_glm.columns)}")
            
            # Run GLM
            # run_glm returns a dict: ch_name -> GLMEstimate object
            glm_estimates_dict: Dict[str, RegressionResults] = run_glm(data_for_glm, design_matrix_for_glm, noise_model=noise_model) # type: ignore # MNE-NIRS stubs might need refinement
            
            contrast_results_dfs: Dict[str, pd.DataFrame] = {}
            active_rois_set = set()
            
            for contrast_name, contrast_weights in contrasts_config.items():
                if not isinstance(contrast_weights, dict): # Double check, should be caught earlier
                     self.logger.warning(f"GLMAnalyzer: Weights for contrast '{contrast_name}' are not a dictionary. Skipping this contrast.")
                     continue
                self.logger.info(f"GLMAnalyzer: Computing contrast: {contrast_name} with weights {contrast_weights}")
                
                temp_contrast_dfs_for_channels = []
                for ch_name, glm_est in glm_estimates_dict.items():
                    try:
                        # Ensure all weights are for columns actually in the design matrix
                        valid_contrast_weights = {k: v for k, v in contrast_weights.items() if k in design_matrix_for_glm.columns}
                        if not valid_contrast_weights:
                             self.logger.warning(f"GLMAnalyzer: Skipping contrast '{contrast_name}' for channel {ch_name}: No valid weights found in design_matrix_for_glm columns.")
                             continue

                        contrast_obj_ch = glm_est.compute_contrast(valid_contrast_weights)
                        contrast_df_ch = contrast_obj_ch.to_dataframe()
                        contrast_df_ch['ch_name'] = ch_name # Add channel name for aggregation
                        temp_contrast_dfs_for_channels.append(contrast_df_ch)
                    except Exception as e_ch_contrast:
                        self.logger.warning(f"Could not compute contrast '{contrast_name}' for channel {ch_name}: {e_ch_contrast}")
                
                if not temp_contrast_dfs_for_channels:
                    self.logger.warning(f"GLMAnalyzer: No channel-wise contrast results for '{contrast_name}'.")
                    contrast_results_dfs[contrast_name] = pd.DataFrame()
                    continue

                aggregated_contrast_df = pd.concat(temp_contrast_dfs_for_channels).reset_index(drop=True)
                contrast_results_dfs[contrast_name] = aggregated_contrast_df
                self.logger.info(f"GLMAnalyzer: Contrast '{contrast_name}' computed. Aggregated DataFrame shape: {aggregated_contrast_df.shape}")
                
                # Optional: Identify active ROIs (e.g., for fNIRS EEG guidance)
                if rois_config and activation_p_threshold is not None and roi_trigger_contrast_name:
                    # rois_config, activation_p_threshold, and roi_trigger_contrast_name are validated earlier.
                    # If rois_config became None due to validation failure, this block won't execute.
                    if contrast_name == roi_trigger_contrast_name:
                        for roi_name_cfg, channels_in_roi_cfg in rois_config.items():
                            # Filter contrast results for channels within this ROI
                            roi_specific_contrast_df = aggregated_contrast_df[
                                aggregated_contrast_df['ch_name'].isin(channels_in_roi_cfg)
                            ]
                            if not roi_specific_contrast_df.empty:
                                # Check for significant activation within the ROI
                                significant_activation = roi_specific_contrast_df[
                                    roi_specific_contrast_df['p_value'] < activation_p_threshold
                                ]
                                if not significant_activation.empty:
                                    active_rois_set.add(roi_name_cfg)
                                    self.logger.info(f"GLMAnalyzer: ROI '{roi_name_cfg}' added to active list.")
                                else:
                                     self.logger.debug(f"GLMAnalyzer: ROI '{roi_name_cfg}' did not meet p-value threshold.")
                            else:
                                self.logger.debug(f"GLMAnalyzer: No contrast results for channels in ROI '{roi_name_cfg}'.")

            if rois_config: # Log final active ROIs only if ROI analysis was attempted (and rois_config was valid)
                self.logger.info(f"GLMAnalyzer: Final active ROIs: {list(active_rois_set)}")

            return {
                'glm_estimates': glm_estimates_dict, # This is now a dict of GLMEstimate objects
                'contrast_results': contrast_results_dfs, # Dict of DataFrames
                'active_rois': list(active_rois_set)
            }
        except Exception as e_glm:
            self.logger.error(f"GLMAnalyzer: Critical error during GLM: {e_glm}", exc_info=True)
            return initial_return_state