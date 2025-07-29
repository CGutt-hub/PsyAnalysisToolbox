"""
Artifacts Utility Module
-----------------------
Provides classes and helpers for managing analysis artifacts.
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any, Dict

class ParticipantArtifacts:
    """
    Manages artifacts for a single participant.
    - Stores and retrieves artifacts by key.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, participant_id: str, output_dir: str):
        self.participant_id = participant_id
        self.output_dir = output_dir
        self._artifacts: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"ParticipantArtifacts_{participant_id}")
        self.logger.info("ParticipantArtifacts initialized.")

    def set(self, key: str, value: Any) -> None:
        """
        Sets an artifact value by key.
        """
        self._artifacts[key] = value
        self.logger.debug(f"Set artifact '{key}'.")

    def get(self, key: str) -> Any:
        """
        Gets an artifact value by key.
        """
        return self._artifacts.get(key)

    def has(self, key: str) -> bool:
        """
        Checks if an artifact exists by key.
        """
        return key in self._artifacts

    def keys(self):
        """
        Returns all artifact keys.
        """
        return self._artifacts.keys()

class GroupArtifacts:
    """
    Encapsulates all data, results, and intermediate files for group-level analysis.
    Provides safe get/set, validation, and serialization methods.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._artifacts = {}

    def set(self, key, value):
        self._artifacts[key] = value

    def get(self, key, default=None):
        return self._artifacts.get(key, default)

    def has(self, key):
        return key in self._artifacts

    def keys(self):
        return list(self._artifacts.keys())

    def validate(self, required_keys):
        missing = [k for k in required_keys if k not in self._artifacts]
        return missing

    def serialize(self, path=None):
        import pickle
        path = path or f"{self.output_dir}/group_artifacts.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self._artifacts, f)
        return path

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self._artifacts = pickle.load(f) 