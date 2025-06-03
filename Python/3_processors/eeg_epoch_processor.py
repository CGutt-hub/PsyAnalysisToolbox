import mne
import numpy as np
from typing import Dict, Optional, Any, List # Added List for picks

class EEGEpochProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGEpochProcessor initialized.")

    def create_epochs(self,
                      raw_processed: mne.io.Raw,
                      events: np.ndarray,
                      event_id: Dict[str, int],
                      tmin: float,
                      tmax: float,
                      picks: Optional[List[str]] = None, # Changed to List[str]
                      baseline: Optional[tuple] = (None, 0),
                      reject: Optional[Dict[str, float]] = None,
                      preload: bool = True,
                      decim: int = 1,
                      reject_by_annotation: bool = True,
                      **kwargs: Any # To pass any other valid mne.Epochs parameters
                      ) -> Optional[mne.Epochs]:
        """
        Creates epochs from processed raw EEG data.

        Args:
            raw_processed (mne.io.Raw): The processed MNE Raw object.
            events (np.ndarray): Events array (n_events, 3).
            event_id (Dict[str, int]): Dictionary mapping event descriptions to event IDs.
            tmin (float): Start time of the epoch relative to the event, in seconds.
            tmax (float): End time of the epoch relative to the event, in seconds.
            picks (Optional[List[str]]): Channels to include. If None, all data channels are used.
            baseline (Optional[tuple]): The time interval to apply baseline correction.
            reject (Optional[Dict[str, float]]): Rejection parameters.
            preload (bool): If True, preload the data into memory.
            decim (int): Factor to downsample the data.
            reject_by_annotation (bool): Whether to reject based on annotations.
            **kwargs: Additional keyword arguments to pass to mne.Epochs.

        Returns:
            Optional[mne.Epochs]: The created MNE Epochs object, or None if an error occurs.
        """
        if raw_processed is None:
            self.logger.warning("EEGEpochProcessor - Processed raw data not provided. Skipping epoch creation.")
            return None
        if events is None or not events.size or event_id is None or not event_id:
            self.logger.warning("EEGEpochProcessor - Events or event_id not provided or empty. Skipping epoch creation.")
            return None

        self.logger.info(f"EEGEpochProcessor - Creating epochs: tmin={tmin}, tmax={tmax}, baseline={baseline}")
        try:
            epochs = mne.Epochs(raw_processed, events, event_id=event_id, tmin=tmin, tmax=tmax,
                                picks=picks, baseline=baseline, reject=reject, preload=preload,
                                decim=decim, reject_by_annotation=reject_by_annotation, verbose=False, **kwargs)
            if len(epochs) == 0:
                self.logger.info("EEGEpochProcessor - No epochs were created (e.g., all events dropped or no events found for given IDs).")
            else:
                self.logger.info(f"EEGEpochProcessor - Created {len(epochs)} epochs.")
            return epochs
        except Exception as e:
            self.logger.error(f"EEGEpochProcessor - Error creating epochs: {e}", exc_info=True)
            return None