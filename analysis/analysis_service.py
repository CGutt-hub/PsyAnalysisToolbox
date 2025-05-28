# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\analysis\analysis_service.py
import numpy as np
import pandas as pd
# import mne_nirs # Not directly used here anymore for GLM
# from mne_nirs.statistics import run_glm # Not directly used here
# from mne.stats import make_first_level_design_matrix # Not directly used here
# Import specialized analyzers
from .hrv_analyzer import HRVAnalyzer
from .psd_analyzer import PSDAnalyzer
from .fnirs_glm_analyzer import FNIRSGLMAnalyzer 
from .connectivity_analyzer import ConnectivityAnalyzer 
from .anova_analyzer import ANOVAAnalyzer
from .eda_analyzer import EDAAnalyzer # Import EDAAnalyzer
from .correlation_analyzer import CorrelationAnalyzer

class AnalysisService:
    def __init__(self, logger, main_config=None): # Accept main_config, though specific configs are preferred for methods
        self.logger = logger
        self.main_config = main_config # Store if needed for any direct use or defaults
        # Instantiate specialized analyzers
        self.hrv_analyzer = HRVAnalyzer(logger)
        self.psd_analyzer = PSDAnalyzer(logger)
        self.fnirs_glm_analyzer = FNIRSGLMAnalyzer(logger) 
        self.connectivity_analyzer = ConnectivityAnalyzer(logger)
        self.anova_analyzer = ANOVAAnalyzer(logger)
        self.eda_analyzer = EDAAnalyzer(logger) # Instantiate EDAAnalyzer
        self.correlation_analyzer = CorrelationAnalyzer(logger)
        self.logger.info("AnalysisService initialized (delegation mode).")

    # --- HRV Related ---
    def calculate_rmssd_from_nni_array(self, nn_intervals_ms):
        """Delegates RMSSD calculation from an array of NNIs."""
        return self.hrv_analyzer.calculate_rmssd_from_nni_array(nn_intervals_ms)

    def calculate_resting_state_rmssd(self, ecg_signal, ecg_sfreq, ecg_times, 
                                      baseline_start_time_abs, baseline_end_time_abs,
                                      ecg_rpeak_method_config): # Add config
        """Delegates resting-state RMSSD calculation."""
        hrv_metrics = self.hrv_analyzer.calculate_resting_state_hrv(
            ecg_signal, ecg_sfreq, ecg_times, 
            baseline_start_time_abs, baseline_end_time_abs,
            ecg_rpeak_method_config=ecg_rpeak_method_config # Pass config
        )
        return hrv_metrics.get('resting_state_rmssd', np.nan)

    def get_continuous_hrv_signal(self, rpeaks_samples, original_sfreq, target_sfreq):
        """Delegates generation of continuous HRV signal."""
        signal, _ = self.hrv_analyzer.get_continuous_hrv_signal(rpeaks_samples, original_sfreq, target_sfreq)
        return signal


    # --- PSD and FAI Related ---
    def calculate_psd_and_fai(self, raw_eeg_processed, events_array, event_id_map,
                                fai_alpha_band_config, eeg_bands_config_for_beta, 
                                fai_electrode_pairs_config, analysis_epoch_tmax_config): # Add configs
        """Delegates PSD and FAI calculation."""
        return self.psd_analyzer.calculate_psd_and_fai(
            raw_eeg_processed, events_array, event_id_map,
            fai_alpha_band_config=fai_alpha_band_config, # Pass config
            eeg_bands_config_for_beta=eeg_bands_config_for_beta, # Pass config
            fai_electrode_pairs_config=fai_electrode_pairs_config, # Pass config
            analysis_epoch_tmax_config=analysis_epoch_tmax_config # Pass config
        )

    # --- fNIRS GLM Related ---
    def run_fnirs_glm_and_contrasts(self, fnirs_epochs_mne, participant_id, analysis_results_dir,
                                      glm_hrf_model, glm_contrasts_config, 
                                      glm_rois_config, glm_activation_p_threshold): # Add configs
        """Delegates fNIRS GLM analysis on epoched data to FNIRSGLMAnalyzer."""
        # The event_id_map for GLM should come from the epochs object itself or be passed if different
        event_id_map_for_glm = fnirs_epochs_mne.event_id if fnirs_epochs_mne else {}
        
        return self.fnirs_glm_analyzer.run_glm_on_epochs(
            fnirs_epochs_mne=fnirs_epochs_mne,
            event_id_map_for_glm=event_id_map_for_glm,
            participant_id=participant_id,
            analysis_results_dir=analysis_results_dir,
            glm_hrf_model=glm_hrf_model, # Pass config
            glm_contrasts_config=glm_contrasts_config, # Pass config
            glm_rois_config=glm_rois_config, # Pass config
            glm_activation_p_threshold=glm_activation_p_threshold # Pass config
        )

    # --- Connectivity Related ---
    def calculate_trial_plv(self, eeg_epochs, eeg_channels_for_plv, 
                            continuous_hrv_signal, hrv_sfreq,
                            phasic_eda_signal, eda_sfreq,
                            plv_eeg_bands_config, # Add config
                            participant_id, raw_eeg_sfreq_for_event_timing,
                            trial_id_eprime_map=None):
        """Delegates trial-wise PLV calculation."""
        return self.connectivity_analyzer.calculate_trial_plv(
            eeg_epochs, eeg_channels_for_plv,
            continuous_hrv_signal, hrv_sfreq,
            phasic_eda_signal, eda_sfreq,
            plv_eeg_bands_config=plv_eeg_bands_config, # Pass config
            participant_id=participant_id, # Pass as keyword
            raw_eeg_sfreq_for_event_timing=raw_eeg_sfreq_for_event_timing, # Pass as keyword
            trial_id_eprime_map=trial_id_eprime_map # Pass as keyword
        )

    # --- Statistical Tests ---
    def run_repeated_measures_anova(self, data_df, dv, within, subject, effsize="np2"):
        """Delegates Repeated Measures ANOVA."""
        return self.anova_analyzer.perform_rm_anova(data_df, dv, within, subject, 
                                                    effsize=effsize, detailed=True)

    def run_correlation_analysis(self, series1, series2, method='pearson', name1='Series1', name2='Series2'):
        """Delegates correlation analysis."""
        corr_result = self.correlation_analyzer.calculate_correlation(series1, series2, method, name1, name2)
        
        if isinstance(corr_result, pd.DataFrame):
            # If CorrelationAnalyzer returns a DataFrame, assume it's correctly formatted
            return corr_result
        elif isinstance(corr_result, dict):
            # If it's a dictionary, check for essential keys that signify a valid correlation.
            # The exact keys depend on what CorrelationAnalyzer is designed to return.
            # For example, 'r' and 'p-val' are common.
            if 'r' in corr_result and 'p-val' in corr_result: # Adjust keys as per CorrelationAnalyzer's output
                return pd.DataFrame([corr_result])
            else:
                self.logger.warning(f"Correlation analysis returned a dictionary without expected keys (e.g., 'r', 'p-val'): {corr_result}")
                return pd.DataFrame() # Return empty DataFrame for an incomplete/invalid dictionary
        elif corr_result is None:
            self.logger.warning("Correlation analysis returned None, possibly due to invalid input or calculation failure.")
            return pd.DataFrame()
        else:
            self.logger.warning(f"Correlation analysis returned an unexpected type: {type(corr_result)}. Expected DataFrame, dict, or None.")
            return pd.DataFrame()

    # --- EDA Features Related ---
    def calculate_eda_features_per_condition(self, raw_eeg_with_events,
                                             phasic_eda_full_signal_array,
                                             eda_original_sfreq,
                                             stimulus_duration_seconds, # Add config
                                             analysis_metrics_dict):
        """Delegates EDA feature calculation per condition."""
        self.eda_analyzer.calculate_eda_features_per_condition(
            raw_eeg_with_events, 
            phasic_eda_full_signal_array, 
            eda_original_sfreq, 
            stimulus_duration_seconds=stimulus_duration_seconds, # Pass config
            analysis_metrics=analysis_metrics_dict # Ensure correct param name if EDAAnalyzer expects 'analysis_metrics'
        )