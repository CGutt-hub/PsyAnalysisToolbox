class ParticipantArtifacts:
    """
    Encapsulates all data, results, and intermediate files for a participant.
    Provides safe get/set, validation, and serialization methods.
    """
    def __init__(self, participant_id, output_dir):
        self.participant_id = participant_id
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
        path = path or f"{self.output_dir}/{self.participant_id}_artifacts.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self._artifacts, f)
        return path

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self._artifacts = pickle.load(f)

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