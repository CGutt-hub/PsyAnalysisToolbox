import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Callable, List, Any
import pandas as pd
import numpy as np
from PsyAnalysisToolbox.Python.utils.data_conversion import _create_eeg_mne_raw_from_df, _create_fnirs_mne_raw_from_df
import mne

# Module-level default for ParallelTaskRunner
PARALLEL_TASK_RUNNER_DEFAULT_MAX_WORKERS: int = os.cpu_count() or 1 # Default to number of CPUs, or 1 if undetectable

class ParallelTaskRunner:
    
    def __init__(self, 
                 task_function: Callable[[Any], Any], 
                 task_configs: List[Any], 
                 main_logger_name: str,
                 max_workers: int = PARALLEL_TASK_RUNNER_DEFAULT_MAX_WORKERS,
                 thread_name_prefix: str = "TaskRunnerThread"):
        """
        Manages parallel execution of a given task function using ThreadPoolExecutor.

        Args:
            task_function: The function to execute for each task.
                           It should accept a single argument (a dictionary or object)
                           containing all necessary configuration for that task. 
                           It can return any result. If the task_function handles its
                           own errors, it's recommended to return a dictionary 
                           with a 'status' key indicating success or specific error.
            task_configs (List[Any]): A list of configuration objects/dictionaries, one for each task. 
                                      Can be an empty list.
            main_logger_name (str): Name of the main logger to use for the runner's own logging.
            max_workers (int, optional): Maximum number of worker threads. 
                                         Defaults to the number of CPU cores.
            thread_name_prefix (str, optional): A prefix for naming the worker threads, useful for debugging.
                                                Defaults to "TaskRunnerThread".
        """
        if not callable(task_function):
            raise TypeError("task_function must be a callable function.")
        if not isinstance(task_configs, list):
            # Allow empty list, but it must be a list.
            raise TypeError("task_configs must be a list.")
        if not isinstance(main_logger_name, str) or not main_logger_name.strip():
            # Cannot log an error here as logger is not yet initialized.
            # Raising an error is appropriate for critical misconfiguration.
            raise ValueError("main_logger_name must be a non-empty string.")

        self.task_function = task_function
        self.task_configs = task_configs
        self.logger = logging.getLogger(main_logger_name) # Initialize logger once, early
        self.thread_name_prefix = thread_name_prefix
        
        if max_workers <= 0:
            self.logger.warning(
                f"max_workers was initialized with {max_workers}, which is not positive. "
                f"Defaulting to 1 worker to ensure ThreadPoolExecutor can start."
            )
            self.max_workers = 1
        else:
            self.max_workers = max_workers
        self.results: List[Any] = []

    def update_tasks(self, new_task_configs: List[Any]):
        """
        Updates or replaces the list of tasks to be run.
        This is useful for continuous or dynamic modes of operation.
        """
        if not isinstance(new_task_configs, list):
            self.logger.error("Attempted to update tasks with a non-list object.")
            raise TypeError("new_task_configs must be a list.")
        self.task_configs = new_task_configs
        self.logger.info(f"Runner tasks updated. Now have {len(self.task_configs)} tasks queued.")

    def run(self) -> List[Any]:
        self.results = [] # Clear results from any previous run
        self.logger.info(f"Starting parallel execution with up to {self.max_workers} workers for {len(self.task_configs)} tasks.")
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.thread_name_prefix) as executor:
            future_to_config = {
                executor.submit(self.task_function, config): config
                for config in self.task_configs
            }

            for future in as_completed(future_to_config):
                task_config_completed = future_to_config[future]
                try:
                    result = future.result() # This will re-raise exceptions from the task_function
                    self.results.append(result)
                except Exception as exc:
                    # Try to create a more informative identifier for the failed task
                    task_id = "Unknown Task"
                    if isinstance(task_config_completed, dict):
                        # Look for common identifying keys to make logs more readable
                        if 'modality' in task_config_completed:
                            task_id = f"Modality: {task_config_completed['modality']}"
                        elif 'participant_id' in task_config_completed:
                            task_id = f"Participant: {task_config_completed['participant_id']}"
                        else:
                            task_id = f"Config (first 50 chars): {str(task_config_completed)[:50]}"
                    else:
                        task_id = f"Config (first 50 chars): {str(task_config_completed)[:50]}"

                    self.logger.error(f"Task '{task_id}' generated an exception: {exc}", exc_info=True)
                    # The task_function itself should return a dict with status='error'
                    # This catch is for unexpected errors in the future.result() or executor itself.
                    self.results.append({'task_config': str(task_config_completed)[:100], 'status': 'runner_exception', 'error_message': str(exc)})
        
        self.logger.info(f"Parallel execution finished. Collected {len(self.results)} results/statuses.")
        return self.results

class DAGTask:
    def __init__(self, name, func, deps, outputs, args=None, kwargs=None):
        self.name = name
        self.func = func
        self.deps = set(deps)
        self.outputs = set(outputs)
        self.args = args or ()
        self.kwargs = kwargs or {}

class DAGParallelTaskRunner:
    def __init__(self, tasks, max_workers=4, logger=None):
        """
        tasks: list of DAGTask
        max_workers: number of parallel workers
        logger: optional logger
        """
        self.tasks = {t.name: t for t in tasks}
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        self.available = set()
        self.completed = set()
        self.results = {}

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pending = set(self.tasks.keys())
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending or futures:
                # Launch all ready tasks
                ready = [name for name in pending if self.tasks[name].deps <= self.available]
                for name in ready:
                    task = self.tasks[name]
                    self.logger.info(f"Launching task: {name} (deps: {task.deps})")
                    futures[executor.submit(task.func, *task.args, **task.kwargs)] = name
                    pending.remove(name)
                # Wait for any task to finish
                if not futures:
                    break  # No tasks running, avoid deadlock
                for future in as_completed(list(futures)):
                    name = futures.pop(future)
                    try:
                        result = future.result()
                        self.logger.info(f"Task completed: {name}")
                    except Exception as exc:
                        self.logger.error(f"Task '{name}' generated an exception: {exc}", exc_info=True)
                        result = {'status': 'error', 'error_message': str(exc)}
                    self.results[name] = result
                    self.available |= self.tasks[name].outputs
                    self.completed.add(name)
                    break  # Go back to check for new ready tasks
        return self.results

def dispatch_preprocessing_task(task):
    """
    Universal worker for physiological preprocessing tasks (EEG, ECG/HRV, EDA, fNIRS).
    Accepts a task dict with keys: 'modality', 'preprocessor', 'data', 'config', 'logger', 'artifacts'.
    Calls the appropriate preprocessor and updates artifacts.
    """
    modality = task['modality']
    preprocessor = task['preprocessor']
    data = task['data']
    config = task['config']
    logger = task['logger']
    artifacts = task['artifacts']
    try:
        logger.info(f"Preprocessing {modality} data...")
        if modality == 'eeg':
            if not isinstance(data, mne.io.BaseRaw):
                logger.error("EEG data is not an MNE Raw object. Skipping EEG preprocessing.")
                return {'modality': modality, 'result': None}
            from preprocessors.eeg_preprocessor import EEGPreprocessor
            eeg_config = EEGPreprocessor.default_config()
            if hasattr(config, 'get'):
                eeg_config['eeg_filter_band'] = (
                    config.getfloat('EEG', 'eeg_filter_l_freq', fallback=1.0),
                    config.getfloat('EEG', 'eeg_filter_h_freq', fallback=40.0)
                )
                ica_n = config.get('EEG', 'ica_n_components', fallback=eeg_config['ica_n_components'])
                if isinstance(ica_n, str):
                    if ica_n.isdigit():
                        ica_n = int(ica_n)
                    elif ica_n == 'rank':
                        pass  # keep as string
                    else:
                        try:
                            ica_n = float(ica_n)
                        except ValueError:
                            pass  # leave as string, will fail validation if not valid
                eeg_config['ica_n_components'] = ica_n
                eeg_config['ica_random_state'] = config.getint('EEG', 'ica_random_state', fallback=eeg_config['ica_random_state'])
                eeg_config['ica_reject_threshold'] = config.getfloat('EEG', 'ica_reject_threshold', fallback=eeg_config['ica_reject_threshold'])
                eeg_config['ica_method'] = config.get('EEG', 'ica_method', fallback=eeg_config['ica_method'])
                eeg_config['ica_extended'] = config.getboolean('EEG', 'ica_extended', fallback=eeg_config['ica_extended'])
                # Add more keys as needed
            logger.info(f"EEG config for preprocessing: {eeg_config}")
            return preprocessor.process(data, eeg_config)
        elif modality == 'ecg':
            # Expect data to be a DataFrame with 'ecg_signal', 'time_sec', etc.
            if isinstance(data, pd.DataFrame) and 'ecg_signal' in data.columns and 'time_sec' in data.columns:
                ecg_signal = data['ecg_signal']
                # Estimate sampling frequency from 'time_sec'
                time_sec = data['time_sec']
                if len(time_sec) > 1:
                    ecg_sfreq = 1 / np.mean(np.diff(time_sec))
                else:
                    logger.error("ECG data has insufficient time points to estimate sampling frequency.")
                    return { 'modality': modality, 'result': None }
                participant_id = artifacts.get('participant_id', 'unknown')
                output_dir = artifacts.get('output_dir', '.')
                result = preprocessor.preprocess_ecg(ecg_signal, ecg_sfreq, participant_id, output_dir, config.get('ECG', 'ecg_rpeak_method', fallback=None))
            else:
                logger.error("ECG data is not a DataFrame with required columns.")
                result = None
            return { 'modality': modality, 'result': result }
        elif modality == 'eda':
            if isinstance(data, pd.DataFrame) and 'eda_signal' in data.columns and 'time_sec' in data.columns:
                eda_signal = data['eda_signal']
                time_sec = data['time_sec']
                if len(time_sec) > 1:
                    eda_sfreq = 1 / np.mean(np.diff(time_sec))
                else:
                    logger.error("EDA data has insufficient time points to estimate sampling frequency.")
                    return { 'modality': modality, 'result': None }
                participant_id = artifacts.get('participant_id', 'unknown')
                output_dir = artifacts.get('output_dir', '.')
                result = preprocessor.preprocess_eda(eda_signal, eda_sfreq, participant_id, output_dir, config.get('EDA', 'eda_cleaning_method', fallback=None))
            else:
                logger.error("EDA data is not a DataFrame with required columns.")
                result = None
            return { 'modality': modality, 'result': result }
        elif modality == 'fnirs':
            from preprocessors.fnirs_preprocessor import FNIRSPreprocessor
            fnirs_config = FNIRSPreprocessor.default_config()
            if hasattr(config, 'get'):
                fnirs_config['beer_lambert_ppf'] = config.getfloat('FNIRS', 'beer_lambert_ppf', fallback=fnirs_config['beer_lambert_ppf'])
                # Add more keys as needed
            logger.info(f"fNIRS config for preprocessing: {fnirs_config}")
            return preprocessor.preprocess(data, fnirs_config, logger)
        else:
            logger.error(f"Unknown modality '{modality}' for preprocessing.")
            return None
    except Exception as e:
        logger.error(f"Error preprocessing {modality}: {e}")
    return { 'modality': modality, 'result': artifacts.get(f"{modality}_preprocessed") }


def preprocess_physiological_data(config, components, logger, artifacts):
    """
    Universal physiological preprocessing dispatcher for EEG, ECG/HRV, EDA, and fNIRS.
    Runs available preprocessors in parallel using ParallelTaskRunner.
    Updates artifacts with results.
    """
    from .parallel_runner import ParallelTaskRunner
    streams = artifacts.get('xdf_streams', {})
    tasks = []
    # EEG: Only add if streams['eeg'] is an MNE Raw object
    if 'eeg' in streams:
        if isinstance(streams['eeg'], mne.io.BaseRaw):
            tasks.append({
                'modality': 'eeg',
                'preprocessor': components['eeg_preprocessor'],
                'data': streams['eeg'],
                'config': config,
                'logger': logger,
                'artifacts': artifacts
            })
        else:
            logger.error("EEG stream is not an MNE Raw object. Skipping EEG preprocessing.")
    # ECG, EDA, fNIRS as before
    for modality in ['ecg', 'eda']:
        key = f'{modality}_df'
        if key in streams and config.getboolean('ProcessingSwitches', f'process_{modality}', fallback=True):
            tasks.append({
                'modality': modality,
                'preprocessor': components[f'{modality}_preprocessor'],
                'data': streams[key],
                'config': config,
                'logger': logger,
                'artifacts': artifacts
            })
    if 'fnirs_cw_amplitude' in streams and config.getboolean('ProcessingSwitches', 'process_fnirs', fallback=True):
        tasks.append({
            'modality': 'fnirs',
            'preprocessor': components['fnirs_preprocessor'],
            'data': streams['fnirs_cw_amplitude'],
            'config': config,
            'logger': logger,
            'artifacts': artifacts
        })
    if not tasks:
        logger.warning("No physiological preprocessing tasks to run.")
        return
    runner = ParallelTaskRunner(
        task_function=dispatch_preprocessing_task,
        task_configs=tasks,
        main_logger_name=logger.name,
        max_workers=config.getint('Parallel', 'max_workers', fallback=4),
        thread_name_prefix="PhysioPreproc"
    )
    results = runner.run()
    for res in results:
        if res is not None and isinstance(res, dict):
            modality = res.get('modality')
            result = res.get('result')
            if modality and result is not None:
                artifacts[f'{modality}_preprocessed'] = result
                logger.info(f"Preprocessing complete for {modality}, result stored as '{modality}_preprocessed'.")
            else:
                logger.warning(f"Preprocessing for {modality} returned no result.")
        else:
            logger.warning(f"A preprocessing task returned no result or unexpected format.")
    logger.info("Universal physiological preprocessing complete.")