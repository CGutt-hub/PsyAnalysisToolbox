import logging
import mne # For mne.Info type hint
from typing import List, Dict, Set, Any
import pandas as pd # To handle DataFrames

# Default configuration for the EEGfNIRSMapper
# - fnirs_roi_to_eeg_map: Maps fNIRS ROI names (str) to lists of EEG channel names (List[str]).
# - default_eeg_channels_for_plv: A list of EEG channel names (List[str]) to use as a fallback.
# - eeg_channel_selection_strategy: Strategy for selection ('mapping', 'nearest', 'predefined').
DEFAULT_EEG_FNIRS_MAPPER_CONFIG = {
    "fnirs_roi_to_eeg_map": {},
    "default_eeg_channels_for_plv": [], 
    "eeg_channel_selection_strategy": "mapping", # Options: 'mapping', 'nearest', 'predefined'
}

class EEGfNIRSMappingProcessor:
    """
    A processor that selects EEG channels based on fNIRS ROI activity and a defined strategy,
    and can add this selection to a results DataFrame.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the EEGfNIRSMapper.
    
        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected keys:
                - 'fnirs_roi_to_eeg_map' (Dict[str, List[str]]): Mapping from fNIRS ROI names to EEG channel lists.
                - 'default_eeg_channels_for_plv' (List[str]): Default EEG channels to use as a fallback.
                - 'eeg_channel_selection_strategy' (str): Strategy like 'mapping', 'nearest', 'predefined'.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger
        self.config = {**DEFAULT_EEG_FNIRS_MAPPER_CONFIG, **config} # Merge provided config with defaults

        # Validate and set fnirs_roi_to_eeg_map
        fnirs_map_config = self.config.get("fnirs_roi_to_eeg_map")
        if isinstance(fnirs_map_config, dict) and all(isinstance(k, str) and isinstance(v, list) and all(isinstance(ch, str) for ch in v) for k, v in fnirs_map_config.items()):
            self.fnirs_roi_to_eeg_map = fnirs_map_config
        else:
            self.logger.warning(f"EEGfNIRSMappingProcessor: 'fnirs_roi_to_eeg_map' is not a valid Dict[str, List[str]]. Using empty map. Config was: {fnirs_map_config}")
            self.fnirs_roi_to_eeg_map = {}

        # Validate and set default_eeg_channels_for_plv
        default_channels_config = self.config.get("default_eeg_channels_for_plv")
        if isinstance(default_channels_config, list) and all(isinstance(ch, str) for ch in default_channels_config):
            self.default_eeg_channels_for_plv = default_channels_config
        else:
            self.logger.warning(f"EEGfNIRSMappingProcessor: 'default_eeg_channels_for_plv' is not a valid List[str]. Using empty list. Config was: {default_channels_config}")
            self.default_eeg_channels_for_plv = []

        self.strategy = str(self.config.get("eeg_channel_selection_strategy", "mapping")) # Ensure strategy is a string

        self.logger.info(f"EEGfNIRSMappingProcessor initialized with strategy: '{self.strategy}'. "
                         f"Default PLV channels configured: {bool(self.default_eeg_channels_for_plv)}. "
                         f"ROI-to-EEG map configured: {bool(self.fnirs_roi_to_eeg_map)}.")

    def _get_fallback_channels(self, eeg_channel_names_set: Set[str]) -> List[str]:
        """
        Helper to get fallback channels if default_eeg_channels_for_plv is set.
        Args:
            eeg_channel_names_set (Set[str]): A set of available EEG channel names from the data.
        """
        if self.default_eeg_channels_for_plv:
            selected_channels = [ch for ch in self.default_eeg_channels_for_plv if ch in eeg_channel_names_set]
            if not selected_channels:
                self.logger.error("None of the configured default_eeg_channels_for_plv were found in the actual EEG channel list.")
            else:
                self.logger.info(f"Using configured default_eeg_channels_for_plv as fallback: {selected_channels}")
            return selected_channels
        return []

    def select_channels(self, eeg_info: mne.Info, fnirs_active_rois: List[str]) -> List[str]:
        """
        Selects EEG channels based on a list of active fNIRS ROI names.

        Args:
            eeg_info (mne.Info): MNE Info object from the EEG data.
            fnirs_active_rois (List[str]): List of names of fNIRS ROIs identified as active.
        Returns:
            List[str]: List of EEG channel names corresponding to the active fNIRS ROIs.
        """
        if not isinstance(eeg_info, mne.Info):
            self.logger.error("EEGfNIRSMappingProcessor: eeg_info must be an MNE Info object. Cannot select channels.")
            return []
        if not isinstance(fnirs_active_rois, list) or not all(isinstance(roi, str) for roi in fnirs_active_rois):
            self.logger.error("EEGfNIRSMappingProcessor: fnirs_active_rois must be a list of strings. Using empty list for active ROIs.")
            fnirs_active_rois = [] # Treat as no active ROIs
    

        final_selected_channels: List[str] = []
        eeg_channel_names_set = set(eeg_info['ch_names'])
    
        # Case 1: No active fNIRS ROIs provided.
        if not fnirs_active_rois:
            self.logger.warning("No active fNIRS ROIs provided. Attempting to use default EEG channels.")
            final_selected_channels = self._get_fallback_channels(eeg_channel_names_set)
            if not final_selected_channels and not self.default_eeg_channels_for_plv:
                # This means no active ROIs AND no default channels were configured at all.
                self.logger.error("No active fNIRS ROIs and no default_eeg_channels_for_plv configured. Cannot select EEG channels.")
            self.logger.info(f"Final selected EEG channels (due to no active ROIs): {final_selected_channels}")
            return final_selected_channels
    
        # Case 2: Active fNIRS ROIs are provided, proceed with strategy.
        selected_eeg_channels_set: Set[str] = set()
    
        if self.strategy == 'mapping':
            if not self.fnirs_roi_to_eeg_map:
                self.logger.warning("Strategy is 'mapping', but fnirs_roi_to_eeg_map is not configured. Will attempt fallback to default channels.")
                # Fallback will be handled by the common logic below
            else: # Map is configured, try to use it
                self.logger.info(f"Attempting EEG channel selection using 'mapping' strategy with {len(fnirs_active_rois)} active ROIs.")
                for roi_name in fnirs_active_rois:
                    if roi_name in self.fnirs_roi_to_eeg_map:
                        channels_for_roi = [ch for ch in self.fnirs_roi_to_eeg_map[roi_name] if ch in eeg_channel_names_set]
                        if channels_for_roi:
                            selected_eeg_channels_set.update(channels_for_roi)
                            self.logger.debug(f"Mapped fNIRS ROI '{roi_name}' to EEG channels: {channels_for_roi}")
                        else:
                            self.logger.warning(f"For ROI '{roi_name}', none of the mapped EEG channels ({self.fnirs_roi_to_eeg_map[roi_name]}) were found in EEG data.")
                    else:
                        self.logger.warning(f"Active fNIRS ROI '{roi_name}' not found in the configured fnirs_roi_to_eeg_map.")
                
                final_selected_channels = list(selected_eeg_channels_set)
                if not final_selected_channels:
                    self.logger.warning("Strategy 'mapping' (with configured map and active ROIs) resulted in no EEG channels. Will attempt fallback to default channels.")
                # If final_selected_channels has items, common fallback logic below will be skipped for this part.
    
        elif self.strategy == 'predefined':
            self.logger.info("Strategy is 'predefined'. Will use default_eeg_channels_for_plv (ignores fNIRS ROIs).")
            # Fallback (which is the primary mechanism here) will be handled by common logic.
    
        elif self.strategy == 'nearest':
            self.logger.warning("Strategy 'nearest' is not implemented. Will attempt fallback to default channels.")
            # Fallback will be handled by common logic.
    
        else:
            self.logger.error(f"Unknown EEG channel selection strategy: '{self.strategy}'. Will attempt fallback to default channels.")
            # Fallback will be handled by common logic.
    
        # Common fallback logic:
        # This is reached if:
        # - Strategy was 'mapping' but map was not configured.
        # - Strategy was 'mapping', map was configured, but it yielded no channels.
        # - Strategy was 'predefined'.
        # - Strategy was 'nearest'.
        # - Strategy was unknown.
        if not final_selected_channels: # If strategy-specific logic did not populate it
            self.logger.debug(f"Primary selection for strategy '{self.strategy}' did not yield channels or strategy dictates using defaults. Attempting fallback.")
            final_selected_channels = self._get_fallback_channels(eeg_channel_names_set)
    
        # Final check and logging
        if not final_selected_channels and not self.default_eeg_channels_for_plv:
            # This is the critical error: all attempts failed AND no defaults were ever configured.
            self.logger.error(f"EEG channel selection failed. Strategy '{self.strategy}' did not yield channels, and no default_eeg_channels_for_plv were configured for fallback.")
        
        self.logger.info(f"Final selected EEG channels using strategy '{self.strategy}' - {len(final_selected_channels)}: {final_selected_channels}")
        return final_selected_channels

    def map_rois_to_channels_df(self,
                                results_df: pd.DataFrame,
                                eeg_info: mne.Info,
                                active_rois: List[str]) -> pd.DataFrame:
        """
        Selects EEG channels based on active fNIRS ROIs and adds them to a results DataFrame.

        This method takes a DataFrame (e.g., from a previous analysis step),
        selects the relevant EEG channels based on the provided active fNIRS ROIs
        and the configured strategy, and returns the DataFrame with a new column
        containing the list of selected channels. This fits the "processor" pattern
        of taking a DataFrame and returning a modified DataFrame.

        Args:
            results_df (pd.DataFrame): The input DataFrame to which the channel list will be added.
            eeg_info (mne.Info): MNE Info object from the EEG data, used to validate channel names.
            active_rois (List[str]): A list of fNIRS ROI names identified as active.

        Returns:
            pd.DataFrame: The input DataFrame with a new 'selected_eeg_channels' column.
        """
        if not isinstance(results_df, pd.DataFrame):
            self.logger.error("Input 'results_df' is not a pandas DataFrame. Returning it unchanged.")
            return results_df

        # Use the existing selection logic to get the list of channels
        selected_channels = self.select_channels(eeg_info=eeg_info, fnirs_active_rois=active_rois)

        # Add the result as a new column. The list is added as a single object to each row.
        self.logger.info(f"Adding selected EEG channels to DataFrame column 'selected_eeg_channels': {selected_channels}")
        results_df['selected_eeg_channels'] = [selected_channels] * len(results_df)

        return results_df