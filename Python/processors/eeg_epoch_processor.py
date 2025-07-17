import mne
import numpy as np
import pandas as pd  # For DataFrame handling
from typing import Dict, Optional, Any, List, Tuple # Added List for picks and Tuple

class EEGEpochProcessor:
    # Default parameters for epoch creation
    DEFAULT_EPOCH_PICKS: Optional[List[str]] = None # None means all data channels
    DEFAULT_EPOCH_BASELINE: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0) # MNE default
    DEFAULT_EPOCH_REJECT: Optional[Dict[str, float]] = None # No rejection by default
    DEFAULT_EPOCH_PRELOAD = True
    DEFAULT_EPOCH_DECIM = 1
    DEFAULT_EPOCH_REJECT_BY_ANNOTATION = True

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGEpochProcessor initialized.")

    def create_epochs(self,
                      raw_processed: mne.io.Raw,
                      events: np.ndarray,
                      event_id: Dict[str, int],
                      tmin: float,
                      tmax: float,
                      picks: Optional[List[str]] = DEFAULT_EPOCH_PICKS,
                      baseline: Optional[Tuple[Optional[float], Optional[float]]] = DEFAULT_EPOCH_BASELINE,
                      reject: Optional[Dict[str, float]] = DEFAULT_EPOCH_REJECT,
                      preload: bool = DEFAULT_EPOCH_PRELOAD,
                      decim: int = DEFAULT_EPOCH_DECIM,
                      reject_by_annotation: bool = DEFAULT_EPOCH_REJECT_BY_ANNOTATION,
                      **kwargs: Any # To pass any other valid mne.Epochs parameters
                      ) -> Optional[pd.DataFrame]:  # Return a DataFrame
        """
        Creates epochs from processed raw EEG data.

        Args:
            raw_processed (mne.io.Raw): The processed MNE Raw object.
            events (np.ndarray): Events array (n_events, 3).
            event_id (Dict[str, int]): Dictionary mapping event descriptions to event IDs.
            tmin (float): Start time of the epoch relative to the event, in seconds.
            tmax (float): End time of the epoch relative to the event, in seconds.
            picks (Optional[List[str]]): Channels to include. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_PICKS.
            baseline (Optional[tuple]): Baseline correction interval. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_BASELINE.
            reject (Optional[Dict[str, float]]): Rejection parameters. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_REJECT.
            preload (bool): Preload data. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_PRELOAD.
            decim (int): Decimation factor. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_DECIM.
            reject_by_annotation (bool): Reject by annotation. Defaults to EEGEpochProcessor.DEFAULT_EPOCH_REJECT_BY_ANNOTATION.
            **kwargs: Additional keyword arguments to pass to mne.Epochs.

        Returns:
            Optional[pd.DataFrame]: A DataFrame representation of the epochs, where each row is an epoch,
 and columns could include epoch data, event information, etc., or None if an error occurs.
       """
        if raw_processed is None:
            self.logger.warning("EEGEpochProcessor - Processed raw data not provided. Skipping epoch creation.")
            return None
        if events is None or not events.size or event_id is None or not event_id:
            self.logger.warning("EEGEpochProcessor - Events or event_id not provided or empty. Skipping epoch creation.")
            return None
        
        # Validate picks if provided
        if picks is not None:
            invalid_picks = [p for p in picks if p not in raw_processed.ch_names]
            if invalid_picks:
                self.logger.warning(f"EEGEpochProcessor - Invalid channel names in 'picks': {invalid_picks}. These will be ignored by MNE or may cause an error.")
                # MNE will handle this, but logging it here is informative.

        self.logger.info(f"EEGEpochProcessor - Creating epochs: tmin={tmin}, tmax={tmax}, baseline={baseline}")
        try:
            epochs = mne.Epochs(raw_processed, events, event_id=event_id, tmin=tmin, tmax=tmax,
                                picks=picks, baseline=baseline, reject=reject, preload=preload,
                                decim=decim, reject_by_annotation=reject_by_annotation, verbose=False, **kwargs)

            if len(epochs) == 0: # Handle case where no epochs were created.
                self.logger.info("EEGEpochProcessor - No epochs were created.")
                return None

            self.logger.info(f"EEGEpochProcessor - Created {len(epochs)} epochs. Converting to DataFrame.")

            # Convert MNE Epochs to DataFrame
            epochs_data = epochs.get_data()  # epochs_data is (n_epochs, n_channels, n_times)
            if epochs.events is not None and len(epochs.events) > 0:
                event_ids = epochs.events[:, 2]  # Get event IDs for each epoch.
            else:
                self.logger.warning("No events found in the epochs object. Setting event_ids to None.")
                event_ids = None

            epochs_df = pd.DataFrame({
                'epoch_data': list(epochs_data),  # Store 3D data as a list of 2D arrays (one per epoch)
                'event_id': event_ids
            })

            # Add more relevant metadata as columns if needed (e.g., event names)

            return epochs_df

        except Exception as e:
            self.logger.error(f"EEGEpochProcessor - Error creating epochs: {e}", exc_info=True)
            return None