"""
Git Handler Utility Module
-------------------------
Provides helpers for managing Git operations in the analysis pipeline.
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any, List

class GitHandler:
    """
    Provides helpers for managing Git operations in the analysis pipeline.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, logger: logging.Logger, repository_path: str, default_remote_name: str = 'origin'):
        self.logger = logger
        self.repository_path = repository_path
        self.default_remote_name = default_remote_name
        self.logger.info("GitHandler initialized.")

    def commit_and_sync_changes(self, paths_to_add: List[str], commit_message: str, remote_name: str = 'origin') -> None:
        """
        Commits and pushes changes to the Git repository.
        """
        import subprocess
        import os
        remote = remote_name if remote_name is not None else self.default_remote_name
        try:
            for path in paths_to_add:
                subprocess.run(['git', 'add', path], cwd=self.repository_path, check=True)
                self.logger.info(f"GitHandler: Added {path} to git staging area.")
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=self.repository_path, check=True)
            self.logger.info(f"GitHandler: Committed with message: {commit_message}")
            subprocess.run(['git', 'push', remote], cwd=self.repository_path, check=True)
            self.logger.info(f"GitHandler: Pushed to remote: {remote}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GitHandler: Git command failed: {e}")
        except Exception as e:
            self.logger.error(f"GitHandler: Unexpected error: {e}")