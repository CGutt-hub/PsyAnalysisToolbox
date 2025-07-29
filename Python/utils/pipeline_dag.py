"""
Pipeline DAG Utility Module
--------------------------
Defines and manages Directed Acyclic Graphs (DAGs) for pipeline execution.
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any, List, Dict, Callable
import pandas as pd
try:
    from mne.io import Raw, RawArray
except ImportError:
    Raw = RawArray = type(None)

# Import validate_and_resolve_artifact from orchestrator if available
try:
    from EmotiViewPrivate.EV_analysis.EV_orchestrator import validate_and_resolve_artifact
except ImportError:
    validate_and_resolve_artifact = None

class DAGTask:
    """
    Represents a single task in a pipeline DAG.
    - Stores task name, function, dependencies, and outputs.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, name: str, func: Callable, deps: List[str], outputs: List[str]):
        self.name = name
        self.func = func
        self.deps = deps
        self.outputs = outputs

# Helper to get nested artifact safely

def get_nested_artifact(artifacts, *keys):
    if hasattr(artifacts, 'get'):
        obj = artifacts.get(keys[0])
    else:
        obj = artifacts[keys[0]]
    for k in keys[1:]:
        if obj is None or k not in obj:
            return None
        obj = obj[k]
    return obj

# Safe process function for each modality with validation

def safe_process(components, artifacts, config, modality, key):
    def _fn():
        # Determine expected type and columns
        if modality == 'eeg':
            expected_type = (Raw, RawArray)
            required_columns = None
        elif modality == 'ecg':
            expected_type = pd.DataFrame
            required_columns = ['ecg_signal']
        elif modality == 'eda':
            expected_type = pd.DataFrame
            required_columns = ['eda_signal']
        elif modality == 'fnirs':
            expected_type = pd.DataFrame
            required_columns = None
        else:
            expected_type = object
            required_columns = None
        # Validate artifact
        obj = artifacts.get(key)
        if validate_and_resolve_artifact is not None:
            obj = validate_and_resolve_artifact(artifacts, key, expected_type, required_columns, logger=logging, aliases=None)
        else:
            # Fallback: basic check
            if obj is None or (required_columns and isinstance(obj, pd.DataFrame) and not all(col in obj.columns for col in required_columns)):
                logging.warning(f"{modality.upper()} artifact '{key}' missing or invalid. Skipping {modality} preprocessing.")
                return None
        if obj is None:
            logging.warning(f"{modality.upper()} artifact '{key}' missing or invalid. Skipping {modality} preprocessing.")
            return None
        return components[f'{modality}_preprocessor'].process(obj, config)
    return _fn


def build_participant_dag(artifacts: Any, config: Any, components: Any, logger: logging.Logger) -> List[DAGTask]:
    """
    Builds a participant-level DAG for pipeline execution (robust version).
    """
    logger.info("PipelineDAG: Building participant DAG.")
    return [
        # Pass the MNE Raw object ('eeg') to the EEG preprocessor, not the DataFrame ('eeg_df')
        DAGTask('preprocess_eeg', safe_process(components, artifacts, config, 'eeg', 'eeg'), [], ['eeg_preprocessed']),
        DAGTask('preprocess_ecg', safe_process(components, artifacts, config, 'ecg', 'ecg_df'), [], ['ecg_preprocessed']),
        DAGTask('preprocess_eda', safe_process(components, artifacts, config, 'eda', 'eda_df'), [], ['eda_preprocessed']),
        DAGTask('preprocess_fnirs', safe_process(components, artifacts, config, 'fnirs', 'fnirs_od_df'), [], ['fnirs_preprocessed'])
    ] 