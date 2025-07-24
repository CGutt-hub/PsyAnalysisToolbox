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