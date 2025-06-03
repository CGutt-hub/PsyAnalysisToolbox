import pandas as pd
import mne
from mne_nirs.statistics import run_glm
from typing import Dict, List, Optional, Any # For type hinting

# Default parameters for GLM
DEFAULT_GLM_NOISE_MODEL = "ar1"
class GLMAnalyzer:
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
                          noise_model: str = DEFAULT_GLM_NOISE_MODEL
                          ) -> Dict[str, Any]:
        """
        Runs a first-level GLM on prepared data using a prepared design matrix and computes specified contrasts.
        Optionally performs ROI-based analysis if ROI configurations are provided (typically for fNIRS).

        Args:
            data_for_glm (mne.io.BaseRaw): Data ready for GLM (e.g., mne.io.RawArray of concatenated evoked responses).
            design_matrix_prepared (pd.DataFrame): The pre-computed design matrix.
            participant_id (str): The ID of the participant.
            contrasts_config (dict): Dictionary defining contrasts based on columns in design_matrix_prepared.
            rois_config (Optional[dict]): Dictionary defining ROIs and their channels (e.g., for fNIRS).
            activation_p_threshold (Optional[float]): P-value threshold for identifying active channels/ROIs (e.g., for fNIRS).
            roi_trigger_contrast_name (Optional[str]): The specific contrast name that should trigger ROI analysis. If None, ROI analysis is not triggered by p-value.
            noise_model (str): The noise model to use for the GLM. Defaults to GLMAnalyzer.DEFAULT_GLM_NOISE_MODEL.

        Returns:
            Dict[str, Any]: A dictionary containing 'glm_estimates' (dict of GLMEstimate objects),
                  'contrast_results' (dict of DataFrames), and optionally 'active_rois' (list).
                  Returns empty dict on failure.
        """
        if data_for_glm is None or design_matrix_prepared is None or design_matrix_prepared.empty:
            self.logger.warning("GLMAnalyzer: Missing data_for_glm or design_matrix_prepared. Skipping GLM.")
            return {'active_rois': [], 'contrast_results': {}, 'glm_estimates': None}
        
        self.logger.info(f"GLMAnalyzer: Running GLM for P:{participant_id}")
        
        if not isinstance(contrasts_config, dict):
            self.logger.error("GLMAnalyzer: contrasts_config must be a dictionary.")
            return {'active_rois': [], 'contrast_results': {}, 'glm_estimates': None}


        # Validate that conditions in contrasts exist in the prepared design matrix columns
        design_matrix_columns = set(design_matrix_prepared.columns)
        for contrast_name, contrast_weights in contrasts_config.items():
            for cond_in_contrast in contrast_weights.keys():
                if cond_in_contrast not in design_matrix_columns:
                    self.logger.warning(f"GLMAnalyzer: Condition '{cond_in_contrast}' in contrast '{contrast_name}' not found in design_matrix_prepared columns. This may cause issues.")

        try:
            # Filter the provided design_matrix_prepared to only include conditions relevant to the
            # specified contrasts, plus standard nuisance regressors (intercept, drifts).
            all_contrast_conditions = set()
            for _, contrast_weights_dict in contrasts_config.items():
                all_contrast_conditions.update(contrast_weights_dict.keys())
            
            design_matrix_cols_to_keep = [
                col for col in design_matrix_prepared.columns
                if col in all_contrast_conditions or col.lower() == 'intercept' or 'drift' in col.lower() or 'constant' in col.lower()
            ]
            design_matrix_for_glm = design_matrix_prepared[design_matrix_cols_to_keep]

            if design_matrix_for_glm.empty or not any(cond in design_matrix_for_glm.columns for cond in all_contrast_conditions):
                 self.logger.warning(f"GLMAnalyzer: Design matrix empty or missing contrast conditions after filtering. Original design cols: {list(design_matrix_prepared.columns)}, Filtered for GLM: {list(design_matrix_for_glm.columns)}. Skipping GLM.")
                 return {'active_rois': [], 'contrast_results': {}, 'glm_estimates': None}

            self.logger.info(f"GLMAnalyzer: Using design matrix with columns for GLM: {list(design_matrix_for_glm.columns)}")
            
            # Run GLM
            glm_estimates_dict = run_glm(data_for_glm, design_matrix_for_glm, noise_model=noise_model)
            # glm_estimates_dict will be a dict: ch_name -> GLMEstimate object
            
            contrast_results_dfs = {}
            active_rois_set = set()
            
            for contrast_name, contrast_weights in contrasts_config.items():
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
                    # rois_config is guaranteed not to be None here due to the outer if condition.
                    if not isinstance(rois_config, dict): 
                        self.logger.warning("GLMAnalyzer: rois_config is not a dictionary. Skipping ROI analysis.")
                    elif contrast_name == roi_trigger_contrast_name: # Use the parameter
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

            if rois_config:
                self.logger.info(f"GLMAnalyzer: Final active ROIs: {list(active_rois_set)}")

            return {
                'glm_estimates': glm_estimates_dict, # This is now a dict of GLMEstimate objects
                'contrast_results': contrast_results_dfs, # Dict of DataFrames
                'active_rois': list(active_rois_set)
            }
        except Exception as e_glm:
            self.logger.error(f"GLMAnalyzer: Critical error during GLM: {e_glm}", exc_info=True)
            return {'active_rois': [], 'contrast_results': {}, 'glm_estimates': None}