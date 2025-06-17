import logging
import pandas as pd # For pd.DataFrame type hint
import numpy as np # For np.ndarray type hint
from typing import Dict, Tuple, Optional, Any


# Default configuration for the MNEEventHandler
# - conditions_to_map: List of string condition names to map to integer IDs.
DEFAULT_MNE_EVENT_HANDLER_CONFIG = {
    "conditions_to_map": [],
}

class MNEEventHandler:
    """
    Creates MNE-compatible event arrays and mappings from an events DataFrame.
    Handles mapping string conditions and trial identifiers to integer IDs.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the MNEEventHandler.

        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected keys:
                - 'conditions_to_map' (List[str]): List of string condition names to map to integer IDs.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.config = {**DEFAULT_MNE_EVENT_HANDLER_CONFIG, **config} # Merge provided config with defaults

        self.conditions_to_map = self.config.get("conditions_to_map", [])
        if not self.conditions_to_map:
             self.logger.warning("MNEEventHandler initialized with an empty 'conditions_to_map'. No conditions will be mapped.")

        self.logger.info(f"MNEEventHandler initialized. Will map {len(self.conditions_to_map)} conditions.")

    def create_events(self, 
                      events_df: pd.DataFrame, 
                      sfreq: float) -> Tuple[Optional[np.ndarray], Dict[str, int], Dict[str, int]]:
        """
        Creates MNE-compatible event arrays and mappings from an events DataFrame.
        Uses the 'conditions_to_map' configured during initialization.

        Args:
            events_df (pd.DataFrame): DataFrame with event information.
                                    Must contain 'onset_time_sec' (or 'onset_sample'),
                                    'condition', and optionally 'trial_identifier_eprime'.
                                    'onset_time_sec' requires sfreq for conversion.
                                    'onset_sample' is preferred if available and sfreq is valid.
            sfreq (float): Sampling frequency to convert 'onset_time_sec' to 'onset_sample' if needed.

        Returns:
            tuple: (mne_events_array, event_id_map, trial_id_eprime_map)
                   Returns (None, {}, {}) if critical errors occur or no events are processable.
        """
        if events_df is None or events_df.empty:
            self.logger.warning("Input events_df is None or empty. Cannot create MNE events.")
            return None, {}, {}

        # Ensure 'onset_sample' is available
        if 'onset_sample' not in events_df.columns:
            if 'onset_time_sec' in events_df.columns:
                if sfreq is not None:
                    if sfreq > 0:
                        events_df['onset_sample'] = (events_df['onset_time_sec'] * sfreq).astype(int)
                        self.logger.info(f"Converted 'onset_time_sec' to 'onset_sample' using sfreq={sfreq}.")
                    elif sfreq == 0:
                        self.logger.error(f"Cannot convert 'onset_time_sec' to 'onset_sample': sfreq is zero, which is invalid for conversion.")
                        return None, {}, {}
                    else: # sfreq < 0
                        self.logger.error(f"Cannot convert 'onset_time_sec' to 'onset_sample': sfreq is negative ({sfreq}), which is invalid.")
                        return None, {}, {}
                else: # sfreq is None
                    self.logger.error("Cannot convert 'onset_time_sec' to 'onset_sample': sfreq is None.")
                    return None, {}, {}
            else:
                self.logger.error("Cannot determine 'onset_sample' for events. DataFrame is missing both 'onset_sample' and 'onset_time_sec' columns.")
                return None, {}, {}

        # Map conditions to integer IDs
        event_id_map = {name: i + 1 for i, name in enumerate(self.conditions_to_map)}
        if not self.conditions_to_map: # Already warned in __init__, but good to be explicit here too
            self.logger.warning("No conditions were configured for mapping in 'conditions_to_map'. Event IDs will likely be 0.")
        
        if 'condition' not in events_df.columns:
            self.logger.error("Events DataFrame missing 'condition' column for MNE event creation.")
            return None, {}, {}

        # Map trial identifiers to integer IDs (optional)
        trial_id_eprime_map = {}
        if 'trial_identifier_eprime' not in events_df.columns:
            self.logger.warning("Events DataFrame missing 'trial_identifier_eprime' column. Numeric trial ID will be 0.")
            events_df['trial_identifier_eprime_numeric'] = 0
        else:        
            # Create a mapping for unique trial identifiers
            # Start numeric IDs high to avoid conflict with condition IDs (typically 1-999)
            unique_trial_ids_eprime = events_df['trial_identifier_eprime'].dropna().unique()
            trial_id_eprime_map = {name: i + 1000 for i, name in enumerate(unique_trial_ids_eprime)} 
            events_df['trial_identifier_eprime_numeric'] = events_df['trial_identifier_eprime'].map(trial_id_eprime_map).fillna(0).astype(int)
            if trial_id_eprime_map:
                self.logger.info(f"Mapped {len(trial_id_eprime_map)} unique trial identifiers.")
            else:
                self.logger.info("No unique trial identifiers found to map in 'trial_identifier_eprime' column or column was all NaNs.")

        # Apply condition mapping and filter for conditions that were mapped
        events_df['condition_id'] = events_df['condition'].map(event_id_map).fillna(0).astype(int)
        
        # Select only rows where condition_id is > 0 (i.e., was successfully mapped)
        # and select the required columns in the correct MNE order
        # MNE events array structure: [sample, previous_event_id, event_id]
        # Here, 'condition_id' serves as the MNE event_id.
        mne_events_sub_df = events_df[events_df['condition_id'] > 0][['onset_sample', 'condition_id']].copy()
        
        if mne_events_sub_df.empty:
             self.logger.warning("After mapping conditions, no events remained with a valid condition ID (> 0). Returning empty events array.")
             return np.array([]).reshape(0, 3), event_id_map, trial_id_eprime_map

        # Insert the 'previous event ID' column (always 0 for standard MNE events)
        mne_events_sub_df.insert(1, 'prev_event_id', 0) 
        # Convert to numpy array
        mne_events_array = mne_events_sub_df.values.astype(int)
        
        self.logger.info(f"Successfully created MNE events array with {len(mne_events_array)} events.")
        return mne_events_array, event_id_map, trial_id_eprime_map