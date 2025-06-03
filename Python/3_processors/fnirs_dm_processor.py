import pandas as pd
import numpy as np
import mne
from mne_nirs.experimental_design import make_first_level_design_matrix
from typing import Dict, List, Optional, Any

class FNIRSDesignMatrixProcessor:
    def __init__(self, logger,
                 hrf_model_config: str = 'glover',
                 drift_model_config: str = 'polynomial',
                 drift_order_config: int = 1,
                 fir_delays_config: Optional[List[float]] = None,
                 oversampling_config: int = 50):
        """
        Initializes the FNIRSDesignMatrixProcessor.

        Args:
            logger: Logger object.
            hrf_model_config (str): HRF model for the GLM (e.g., 'glover', 'spm', 'fir').
            drift_model_config (str): Drift model (e.g., 'polynomial', 'cosine').
            drift_order_config (int): Order of the drift model (for polynomial).
            fir_delays_config (Optional[List[float]]): Delays for FIR model, if hrf_model is 'fir'.
            oversampling_config (int): Oversampling factor for HRF convolution.
        """
        self.logger = logger
        self.hrf_model_config = hrf_model_config
        self.drift_model_config = drift_model_config
        self.drift_order_config = drift_order_config
        self.fir_delays_config = fir_delays_config
        self.oversampling_config = oversampling_config
        self.logger.info("FNIRSDesignMatrixProcessor initialized.")

    def create_design_matrix(self,
                             participant_id: str,
                             xdf_markers_df: pd.DataFrame,
                             raw_fnirs_data: mne.io.BaseRaw,
                             fnirs_stream_start_time_xdf: float,
                             event_mapping_config: Dict[Any, str],
                             condition_duration_config: Dict[str, float],
                             custom_regressors_df: Optional[pd.DataFrame] = None
                             ) -> Optional[pd.DataFrame]:
        """
        Creates a first-level design matrix for fNIRS GLM analysis.

        Args:
            participant_id (str): Identifier for the participant.
            xdf_markers_df (pd.DataFrame): DataFrame of raw XDF markers with 'timestamp' and 'marker_value'.
            raw_fnirs_data (mne.io.BaseRaw): The raw fNIRS data object (e.g., from XDFReader after preproc).
                                             Used to get sfreq and n_times.
            fnirs_stream_start_time_xdf (float): Absolute XDF timestamp of the first fNIRS data sample.
            event_mapping_config (Dict[Any, str]): Maps raw marker values to condition/trial_type names.
                                                   (e.g., {101: 'ConditionA', 'S 12': 'ConditionB'})
            condition_duration_config (Dict[str, float]): Maps condition names to their durations in seconds.
                                                          (e.g., {'ConditionA': 5.0, 'ConditionB': 10.0})
            custom_regressors_df (Optional[pd.DataFrame]): DataFrame of custom regressors to add.
                                                           Must have the same number of rows as the fNIRS data.

        Returns:
            Optional[pd.DataFrame]: The generated design matrix, or None if an error occurs.
        """
        self.logger.info(f"FNIRSDesignMatrixProcessor: Creating design matrix for P:{participant_id}.")

        if xdf_markers_df is None or xdf_markers_df.empty:
            self.logger.warning("FNIRSDesignMatrixProcessor: xdf_markers_df is empty or None. Cannot create events.")
            return None
        if raw_fnirs_data is None:
            self.logger.warning("FNIRSDesignMatrixProcessor: raw_fnirs_data is None. Cannot get timing info.")
            return None
        if fnirs_stream_start_time_xdf is None:
            self.logger.warning("FNIRSDesignMatrixProcessor: fnirs_stream_start_time_xdf is None. Cannot align markers.")
            return None
        if not event_mapping_config:
            self.logger.warning("FNIRSDesignMatrixProcessor: event_mapping_config is empty. Cannot map markers to conditions.")
            return None
        if not condition_duration_config:
            self.logger.warning("FNIRSDesignMatrixProcessor: condition_duration_config is empty. Cannot set event durations.")
            return None

        try:
            fnirs_sfreq = raw_fnirs_data.info['sfreq']
            fnirs_n_times = raw_fnirs_data.n_times
            frame_times = np.arange(fnirs_n_times) / fnirs_sfreq

            # Prepare events DataFrame for make_first_level_design_matrix
            events_list = []
            for _, row in xdf_markers_df.iterrows():
                marker_val = row['marker_value']
                if marker_val in event_mapping_config:
                    condition_name = event_mapping_config[marker_val]
                    if condition_name in condition_duration_config:
                        onset_time_sec = row['timestamp'] - fnirs_stream_start_time_xdf
                        duration_sec = condition_duration_config[condition_name]
                        
                        # Ensure onset is not negative (marker before fNIRS data start)
                        # and not beyond the fNIRS data duration
                        if 0 <= onset_time_sec < (fnirs_n_times / fnirs_sfreq):
                            events_list.append({
                                'trial_type': condition_name,
                                'onset': onset_time_sec,
                                'duration': duration_sec
                            })
                        else:
                            self.logger.debug(f"FNIRSDesignMatrixProcessor: Marker '{marker_val}' at XDF time {row['timestamp']} (onset {onset_time_sec:.2f}s) is outside fNIRS data time range [0, {fnirs_n_times / fnirs_sfreq:.2f}s]. Skipping.")
                    else:
                        self.logger.debug(f"FNIRSDesignMatrixProcessor: Duration for condition '{condition_name}' (from marker '{marker_val}') not found in condition_duration_config. Skipping.")
                else:
                    self.logger.debug(f"FNIRSDesignMatrixProcessor: Marker value '{marker_val}' not found in event_mapping_config. Skipping.")
            
            if not events_list:
                self.logger.warning("FNIRSDesignMatrixProcessor: No valid events created after mapping and filtering. Cannot generate design matrix.")
                return None
            
            events_df = pd.DataFrame(events_list)
            self.logger.info(f"FNIRSDesignMatrixProcessor: Created {len(events_df)} events for design matrix.")

            design_matrix = make_first_level_design_matrix(
                frame_times, events_df,
                hrf_model=self.hrf_model_config, drift_model=self.drift_model_config,
                drift_order=self.drift_order_config, fir_delays=self.fir_delays_config,
                oversampling=self.oversampling_config
            )

            if custom_regressors_df is not None:
                if len(custom_regressors_df) == len(design_matrix):
                    design_matrix = pd.concat([design_matrix, custom_regressors_df.reset_index(drop=True)], axis=1)
                    self.logger.info(f"FNIRSDesignMatrixProcessor: Concatenated custom regressors. New shape: {design_matrix.shape}")
                else:
                    self.logger.warning("FNIRSDesignMatrixProcessor: Custom regressors length mismatch. Not concatenated.")
            
            self.logger.info(f"FNIRSDesignMatrixProcessor: Design matrix created successfully. Shape: {design_matrix.shape}, Columns: {list(design_matrix.columns)}")
            return design_matrix

        except Exception as e:
            self.logger.error(f"FNIRSDesignMatrixProcessor: Error creating design matrix for P:{participant_id}: {e}", exc_info=True)
            return None