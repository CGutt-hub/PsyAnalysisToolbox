import logging
import os

class LogReporter:
    """
    Manages a dedicated logger for a single participant, writing to a specific file.
    """
    # Class-level defaults
    DEFAULT_LOG_LEVEL_STR = "INFO"
    DEFAULT_LOG_FORMATTER = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    DEFAULT_LOG_FILENAME_TEMPLATE = "{participant_id}_processing.log"

    def __init__(self, participant_base_output_dir: str, 
                 participant_id: str, 
                 log_level_str: str = DEFAULT_LOG_LEVEL_STR): # Use class default
        """
        Initializes the participant-specific logger.

        Args:
            participant_base_output_dir (str): The base output directory for this participant.
            participant_id (str): The ID of the participant.
            log_level_str (str): The desired logging level string (e.g., 'DEBUG', 'INFO').
                                 Defaults to LogReporter.DEFAULT_LOG_LEVEL_STR.
        """
        self.participant_id = participant_id
        self.log_file_path = os.path.join(participant_base_output_dir, self.DEFAULT_LOG_FILENAME_TEMPLATE.format(participant_id=participant_id))
        self._logger = logging.getLogger(f"ParticipantLogger_{participant_id}")
        self._handler = None # To store the file handler

        # Prevent adding multiple handlers if the logger is reused (e.g., in interactive sessions)
        if not self._logger.handlers:
            # Set logger level (should be lowest level you want to process)
            self._logger.setLevel(logging.DEBUG) # Set logger to DEBUG to capture all messages

            # Create file handler
            try:
                # Ensure the directory exists before creating the file
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                self._handler = logging.FileHandler(self.log_file_path, mode='w')
                # Set handler level based on provided or default log_level_str
                log_level = getattr(logging, log_level_str.upper(), getattr(logging, self.DEFAULT_LOG_LEVEL_STR.upper()))
                self._handler.setLevel(log_level)

                # Create formatter
                formatter = logging.Formatter(self.DEFAULT_LOG_FORMATTER)
                self._handler.setFormatter(formatter)

                # Add handler to logger
                self._logger.addHandler(self._handler)
                self._logger.info(f"ParticipantLogger initialized for {participant_id}. Logging to {self.log_file_path} at level {log_level_str}.")
            except Exception as e:
                print(f"CRITICAL ERROR (LogReporter): Failed to create file handler for participant {participant_id} at {self.log_file_path}: {e}. Participant-specific logs will NOT be saved to this file. They may appear in the main orchestrator log if configured, or be lost if not.")
                self._handler = None # Ensure handler is None if creation failed


    def get_logger(self):
        """
        Returns the configured logger instance for the participant.
        """
        return self._logger

    def close_handlers(self):
        """
        Closes the file handler associated with this logger.
        Should be called when processing for the participant is complete.
        """
        if self._handler:
            try:
                # Check if the handler is still attached before removing
                if self._handler in self._logger.handlers:
                     self._logger.removeHandler(self._handler)
                self._handler.close()
                self._handler = None # Mark handler as closed and removed
                # Note: We don't log here because the handler is being closed.
                # The main orchestrator log should indicate participant processing is finished.
            except Exception as e:
                # Log to the main logger if closing fails, as participant logger might be unusable
                # We can't rely on the participant logger here, so print or use a known main logger
                main_logger = logging.getLogger("MainOrchestrator") # Assuming this logger exists
                # Check if the main_logger actually exists and has handlers to avoid issues
                if main_logger and main_logger.hasHandlers():
                    main_logger.error(f"Error closing log handler for participant {self.participant_id}: {e}", exc_info=True)
                else:
                    print(f"ERROR: Error closing log handler for participant {self.participant_id}: {e}")
                # Even if closing failed, subsequent calls to close_handlers should not re-attempt on this instance
                self._handler = None