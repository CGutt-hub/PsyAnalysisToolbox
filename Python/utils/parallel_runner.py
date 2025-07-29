"""
Parallel Runner Utility Module
-----------------------------
Provides helpers for running tasks in parallel (thread/process pools).
Config-driven, robust, and maintainable.
"""
import logging
from typing import Any, Callable, List, Dict

class ParallelTaskRunner:
    """
    Runs tasks in parallel using thread or process pools.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, task_function: Callable, task_configs: List[Dict[str, Any]], main_logger_name: str, max_workers: int = 4, thread_name_prefix: str = "Worker"):
        self.task_function = task_function
        self.task_configs = task_configs
        self.main_logger_name = main_logger_name
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.logger = logging.getLogger(main_logger_name)
        self.logger.info("ParallelTaskRunner initialized.")

    def run(self) -> List[Any]:
        """
        Runs all tasks in parallel and returns their results.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(self.task_configs)
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.thread_name_prefix) as executor:
            future_to_idx = {executor.submit(self.task_function, config): idx for idx, config in enumerate(self.task_configs)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"ParallelTaskRunner: Task {idx} failed: {e}", exc_info=True)
                    results[idx] = None
        return results

class DAGParallelTaskRunner(ParallelTaskRunner):
    """
    Runs tasks in parallel according to a Directed Acyclic Graph (DAG) of dependencies.
    - Usable in any project (no project-specific assumptions).
    """
    def __init__(self, dag_tasks: List[Any], logger: logging.Logger, max_workers: int = 4):
        super().__init__(task_function=(lambda *args, **kwargs: None), task_configs=[], main_logger_name=logger.name, max_workers=max_workers)
        self.dag_tasks = dag_tasks
        self.logger = logger
        self.logger.info("DAGParallelTaskRunner initialized.")

    def run(self) -> List[Any]:
        """
        Runs all DAG tasks in parallel and returns their results.
        Supports both dict-based and object-based (DAGTask) tasks.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        completed = set()
        results = [None] * len(self.dag_tasks)
        task_indices = list(range(len(self.dag_tasks)))
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="DAGWorker") as executor:
            futures = {}
            while len(completed) < len(self.dag_tasks):
                ready = []
                for i in task_indices:
                    if i in completed:
                        continue
                    task = self.dag_tasks[i]
                    # Support both object (DAGTask) and dict
                    if hasattr(task, 'deps'):
                        deps = getattr(task, 'deps', [])
                    else:
                        deps = task.get('deps', [])
                    if all(dep in completed for dep in deps):
                        ready.append(i)
                for i in ready:
                    if i not in futures:
                        task = self.dag_tasks[i]
                        if hasattr(task, 'func'):
                            func = getattr(task, 'func')
                        else:
                            func = task.get('func')
                        futures[i] = executor.submit(func)
                for i, fut in list(futures.items()):
                    if fut.done():
                        try:
                            results[i] = fut.result()
                        except Exception as e:
                            self.logger.error(f"DAGParallelTaskRunner: Task {i} failed: {e}", exc_info=True)
                            results[i] = None
                        completed.add(i)
                        del futures[i]
        return results