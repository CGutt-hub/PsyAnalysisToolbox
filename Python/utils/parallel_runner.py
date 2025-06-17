import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Callable, List, Any

# Module-level default for ParallelTaskRunner
PARALLEL_TASK_RUNNER_DEFAULT_MAX_WORKERS: int = os.cpu_count() or 1 # Default to number of CPUs, or 1 if undetectable

class ParallelTaskRunner:
    
    def __init__(self, 
                 task_function: Callable[[Any], Any], 
                 task_configs: List[Any], 
                 main_logger_name: str,
                 max_workers: int = PARALLEL_TASK_RUNNER_DEFAULT_MAX_WORKERS):
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
                                         Defaults to PARALLEL_TASK_RUNNER_DEFAULT_MAX_WORKERS.
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
        
        if max_workers <= 0:
            self.logger.warning(
                f"max_workers was initialized with {max_workers}, which is not positive. "
                f"Defaulting to 1 worker to ensure ThreadPoolExecutor can start."
            )
            self.max_workers = 1
        else:
            self.max_workers = max_workers
        self.results: List[Any] = []

    def run(self) -> List[Any]:
        self.results = [] # Clear results from any previous run
        self.logger.info(f"Starting parallel execution with up to {self.max_workers} workers for {len(self.task_configs)} tasks.")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                    self.logger.error(f"Task with config (first 100 chars): '{str(task_config_completed)[:100]}' generated an exception: {exc}", exc_info=True)
                    # The task_function itself should return a dict with status='error'
                    # This catch is for unexpected errors in the future.result() or executor itself.
                    self.results.append({'task_config': str(task_config_completed)[:100], 'status': 'runner_exception', 'error_message': str(exc)})
        
        self.logger.info(f"Parallel execution finished. Collected {len(self.results)} results/statuses.")
        return self.results