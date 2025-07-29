"""
Log Reporter Module
------------------
Handles logging for reporting and diagnostics.
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any

class LogReporter:
    """
    Handles logging for reporting and diagnostics.
    - Accepts config dict for logging parameters.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, participant_base_output_dir: str, participant_id: str, log_level_str: str):
        self.participant_base_output_dir = participant_base_output_dir
        self.participant_id = participant_id
        self.log_level_str = log_level_str
        self.logger = logging.getLogger(f"LogReporter_{participant_id}")
        self.logger.setLevel(getattr(logging, log_level_str.upper(), logging.INFO))
        self.logger.info("LogReporter initialized.")

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger instance for this reporter.
        """
        return self.logger