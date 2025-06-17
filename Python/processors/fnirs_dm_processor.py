import pandas as pd
import numpy as np
import mne
from mne_nirs.experimental_design import make_first_level_design_matrix
from typing import Dict, List, Optional, Any

class FNIRSDesignMatrixProcessor:
    # Default parameters for design matrix creation
    DEFAULT_HRF_MODEL = 'glover'
    DEFAULT_DRIFT_MODEL = 'polynomial'
    DEFAULT_DRIFT_ORDER = 1
    DEFAULT_FIR_DELAYS: Optional[List[float]] = None # MNE-NIRS default is [0, 1, ..., 5] if FIR is chosen
    DEFAULT_OVERSAMPLING = 50

    def __init__(self, logger,
                 hrf_model_config: str = DEFAULT_HRF_MODEL,
                 drift_model_config: str = DEFAULT_DRIFT_MODEL,
                 drift_order_config: int = DEFAULT_DRIFT_ORDER,
                 fir_delays_config: Optional[List[float]] = DEFAULT_FIR_DELAYS,
                 oversampling_config: int = DEFAULT_OVERSAMPLING):
        """
        Initializes the FNIRSDesignMatrixProcessor.

        Args:
            logger: Logger object.
            hrf_model_config (str): HRF model for the GLM. Defaults to FNIRSDesignMatrixProcessor.DEFAULT_HRF_MODEL.
            drift_model_config (str): Drift model. Defaults to FNIRSDesignMatrixProcessor.DEFAULT_DRIFT_MODEL.
            drift_order_config (int): Order of the drift model. Defaults to FNIRSDesignMatrixProcessor.DEFAULT_DRIFT_ORDER.
            fir_delays_config (Optional[List[float]]): Delays for FIR model. Defaults to FNIRSDesignMatrixProcessor.DEFAULT_FIR_DELAYS.
            oversampling_config (int): Oversampling factor for HRF. Defaults to FNIRSDesignMatrixProcessor.DEFAULT_OVERSAMPLING.
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
            if not (isinstance(fnirs_sfreq, (int, float)) and fnirs_sfreq > 0):
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Invalid sfreq from raw_fnirs_data: {fnirs_sfreq}. Must be a positive number.")
                return None

            fnirs_n_times = raw_fnirs_data.n_times
            if fnirs_n_times <= 0:
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: raw_fnirs_data has no time points (n_times={fnirs_n_times}). Cannot generate frame_times.")
                return None

            frame_times = np.arange(fnirs_n_times, dtype=np.float64) / fnirs_sfreq
            if not np.all(np.isfinite(frame_times)):
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: frame_times contains non-finite values (NaN or Inf). This is unexpected. sfreq={fnirs_sfreq}, n_times={fnirs_n_times}")
                self.logger.debug(f"Problematic frame_times for P:{participant_id}: {frame_times[~np.isfinite(frame_times)]}")
                return None

            # Prepare events DataFrame for make_first_level_design_matrix
            events_list = []
            for _, row in xdf_markers_df.iterrows():
                marker_val = row['marker_value']
                if marker_val in event_mapping_config:
                    condition_name = event_mapping_config[marker_val]
                    if condition_name in condition_duration_config:
                        # Ensure inputs for calculations are numeric and valid
                        try:
                            timestamp_val = float(row['timestamp'])
                            start_time_val = float(fnirs_stream_start_time_xdf)
                            duration_val = float(condition_duration_config[condition_name])
                        except (ValueError, TypeError) as e_conv:
                            self.logger.warning(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Could not convert event timing/duration values to float for marker '{marker_val}'. Error: {e_conv}. Skipping event.")
                            continue

                        onset_time_sec = timestamp_val - start_time_val
                        duration_sec = duration_val

                        # Check for NaN or Inf values after calculation
                        if np.isnan(onset_time_sec) or np.isinf(onset_time_sec) or \
                           np.isnan(duration_sec) or np.isinf(duration_sec) or duration_sec <= 0:
                            self.logger.warning(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Invalid (NaN, Inf, or non-positive) onset/duration for marker '{marker_val}'. Onset: {onset_time_sec}, Duration: {duration_sec}. Skipping event.")
                            continue
                        
                        # Ensure onset is not negative (marker before fNIRS data start)
                        # and not beyond the fNIRS data duration
                        if 0 <= onset_time_sec < (fnirs_n_times / fnirs_sfreq):
                            events_list.append({
                                'trial_type': condition_name,
                                'onset': onset_time_sec, # Already a float
                                'duration': duration_sec # Already a float
                            })
                        else:
                            self.logger.debug(f"FNIRSDesignMatrixProcessor: Marker '{marker_val}' at XDF time {row['timestamp']} (onset {onset_time_sec:.2f}s) is outside fNIRS data time range [0, {fnirs_n_times / fnirs_sfreq:.2f}s]. Skipping.")
                    else:
                        self.logger.debug(f"FNIRSDesignMatrixProcessor: Duration for condition '{condition_name}' (from marker '{marker_val}') not found in condition_duration_config. Skipping.")
                else:
                    self.logger.debug(f"FNIRSDesignMatrixProcessor: Marker value '{marker_val}' not found in event_mapping_config. Skipping.")
            
            if not events_list:
                self.logger.warning(f"FNIRSDesignMatrixProcessor: P:{participant_id}: No valid events in events_list. Initializing an empty events DataFrame for potential drift-only model.")
                # Initialize with correct structure even if events_list is empty
                events_df = pd.DataFrame({
                    'trial_type': pd.Series([], dtype=str),
                    'onset': pd.Series([], dtype=np.float64),
                    'duration': pd.Series([], dtype=np.float64)
                })
            else:
                events_df = pd.DataFrame(events_list)

            # Ensure essential columns exist even if events_df was created from a sparse events_list
            for col_name, default_dtype in [('onset', np.float64), ('duration', np.float64), ('trial_type', str)]:
                if col_name not in events_df.columns:
                    events_df[col_name] = pd.Series([], dtype=default_dtype)

            try:
                events_df['onset'] = pd.to_numeric(events_df['onset'], errors='coerce')
                events_df['duration'] = pd.to_numeric(events_df['duration'], errors='coerce')
            except Exception as e_to_numeric:
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Error during pd.to_numeric conversion of onset/duration: {e_to_numeric}. Design matrix creation likely to fail.", exc_info=True)
                return None # Return early if basic conversion fails

            # Replace any infinite values with NaN so they can be dropped by dropna
            events_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Check for NaNs that might have been introduced by 'coerce', were already present, or came from Inf
            if events_df['onset'].isnull().any() or events_df['duration'].isnull().any():
                nan_onset_count = events_df['onset'].isnull().sum()
                nan_duration_count = events_df['duration'].isnull().sum()
                self.logger.warning(
                    f"FNIRSDesignMatrixProcessor: P:{participant_id}: NaN values detected in 'onset' ({nan_onset_count} NaNs) "
                    f"or 'duration' ({nan_duration_count} NaNs) columns after robust numeric conversion. "
                    f"This may cause issues with MNE-NIRS. Problematic events might be ignored by MNE-NIRS or cause errors."
                )
                self.logger.info(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Dropping rows with NaN in 'onset' or 'duration'.")
                events_df.dropna(subset=['onset', 'duration'], inplace=True)
                if events_df.empty: # If dropna made it empty, re-ensure structure
                    self.logger.warning(f"FNIRSDesignMatrixProcessor: P:{participant_id}: All events removed after dropping NaNs. Re-initializing empty events DataFrame for potential drift-only model.")
                    events_df = pd.DataFrame({
                        'trial_type': pd.Series([], dtype=str),
                        'onset': pd.Series([], dtype=np.float64),
                        'duration': pd.Series([], dtype=np.float64)
                    })
            
            # Ensure trial_type is string, even if events_df was not empty
            if not events_df.empty:
                events_df['trial_type'] = events_df['trial_type'].astype(str)
            # If events_df was re-created as empty, trial_type already has dtype str

            # Final validation of events_df and frame_times before MNE-NIRS call
            required_cols_validation = {
                'onset': pd.api.types.is_float_dtype,
                'duration': pd.api.types.is_float_dtype,
                # For trial_type, MNE-NIRS expects strings.
                # Pandas object dtype often holds strings, or it could be pandas' specific StringDtype.
                'trial_type': lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
            }
            
            for col, validation_func in required_cols_validation.items():
                if col not in events_df.columns:
                    self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Critical - events_df is missing required column '{col}' before MNE call. Columns: {events_df.columns.tolist()}")
                    return None
                if not validation_func(events_df[col]): # Pass the Series to the validation function
                    expected_kind_str = "float" if col in ['onset', 'duration'] else "string or object"
                    self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Critical - events_df column '{col}' has unexpected dtype '{events_df[col].dtype}' (expected: {expected_kind_str}) before MNE call.")
                    return None
            
            # Ensure no NaN/Inf values remain in critical numeric columns
            # This check is a final assertion, especially if events_df was not empty.
            # Convert to NumPy array first, then apply checks.
            onset_duration_values = events_df[['onset', 'duration']].to_numpy()
            condition_has_nan_arr = pd.isnull(onset_duration_values).any(axis=1) # pd.isnull works on numpy arrays
            condition_has_inf_arr = np.isinf(onset_duration_values).any(axis=1)
            problematic_rows_exist_arr = condition_has_nan_arr | condition_has_inf_arr
            
            if not events_df.empty and problematic_rows_exist_arr.any():
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: Critical - NaN or Inf values still present in 'onset' or 'duration' of events_df before MNE call, despite cleaning. This should not happen.")
                self.logger.debug(f"Problematic events_df head for P:{participant_id}:\n{events_df[problematic_rows_exist_arr].head().to_string()}")
                return None
            
            # Final type checks before the call
            if not isinstance(frame_times, np.ndarray) or frame_times.ndim != 1 or not np.issubdtype(frame_times.dtype, np.floating):
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: frame_times is not a 1D float numpy array before MNE call. Type: {type(frame_times)}, Dtype: {frame_times.dtype if hasattr(frame_times, 'dtype') else 'N/A'}")
                return None
            if not isinstance(events_df, pd.DataFrame):
                self.logger.error(f"FNIRSDesignMatrixProcessor: P:{participant_id}: events_df is not a pandas DataFrame before MNE call. Type: {type(events_df)}")
                return None

            self.logger.info(f"FNIRSDesignMatrixProcessor: Created {len(events_df)} events for design matrix.")
            self.logger.debug(f"FNIRSDesignMatrixProcessor: events_df dtypes before MNE call: {events_df.dtypes.to_dict()}")
            self.logger.debug(f"FNIRSDesignMatrixProcessor: events_df head before MNE call:\n{events_df.head().to_string()}")
            self.logger.debug(f"FNIRSDesignMatrixProcessor: frame_times type: {type(frame_times)}, shape: {frame_times.shape if isinstance(frame_times, np.ndarray) else 'N/A'}, dtype: {frame_times.dtype if isinstance(frame_times, np.ndarray) else 'N/A'}")

            # For Pylance's benefit, create explicitly typed variables for the call
            # This helps the static analyzer confirm the types it expects for the MNE function.
            final_frame_times: np.ndarray = frame_times
            final_events_df: pd.DataFrame = events_df

            design_matrix = make_first_level_design_matrix(
                final_frame_times,  # Pass as first positional argument
                final_events_df,    # Pass as second positional argument # type: ignore[call-arg]
                hrf_model=self.hrf_model_config,
                drift_model=self.drift_model_config,
                drift_order=self.drift_order_config,
                fir_delays=self.fir_delays_config,
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