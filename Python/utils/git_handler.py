import subprocess
import os
import logging
from typing import List, Optional

# Module-level defaults for GitHandler
GIT_HANDLER_DEFAULT_REMOTE_NAME: str = "origin"
GIT_HANDLER_DEFAULT_BRANCH_NAME: Optional[str] = None # None means push current branch (HEAD)

class GitHandler:

    def __init__(self, 
                 repository_path: str, 
                 logger: logging.Logger, 
                 default_remote_name: str = GIT_HANDLER_DEFAULT_REMOTE_NAME,
                 default_branch_name: Optional[str] = GIT_HANDLER_DEFAULT_BRANCH_NAME):
        """
        Initializes the GitHandler for a specific repository.

        Args:
            repository_path (str): Absolute path to the root of the Git repository.
            logger (logging.Logger): Logger instance for logging Git operations.
            default_remote_name (str, optional): Default remote name to use for push. Defaults to "origin".
            default_branch_name (str, optional): Default branch name to push. If None, pushes current branch (HEAD). Defaults to None.
        """
        self.repository_path = repository_path
        self.logger = logger
        self.default_remote_name = default_remote_name
        self.default_branch_name = default_branch_name

        if not os.path.isdir(os.path.join(self.repository_path, '.git')):
            self.logger.error(f"Specified repository path is not a Git repository: {self.repository_path}")
            raise ValueError(f"Not a Git repository: {self.repository_path}")
        
        # Proactively check if git command is available and working
        if not self._run_git_command(['git', '--version']):
            # _run_git_command would have logged the specific error (e.g., FileNotFoundError)
            raise RuntimeError("Git command '--version' failed. Ensure Git is installed, in PATH, and the repository is accessible.")

    def _run_git_command(self, command: List[str]) -> bool:
        """Helper function to run a Git command and log its output/errors."""
        self.logger.info(f"Running git command: {' '.join(command)} in {self.repository_path}")
        try:
            process = subprocess.run(command, cwd=self.repository_path, check=True, capture_output=True, text=True, encoding='utf-8')
            if process.stdout:
                self.logger.info(f"Git command stdout:\n{process.stdout}")
            if process.stderr: # Some successful git commands output to stderr (e.g., progress)
                self.logger.info(f"Git command stderr:\n{process.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git command failed: {' '.join(command)}")
            self.logger.error(f"Return code: {e.returncode}")
            if e.stdout:
                self.logger.error(f"Stdout:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"Stderr:\n{e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("Git command not found. Ensure Git is installed and in your system's PATH.")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while running git command: {e}")
            return False

    def commit_and_sync_changes(self,
                                paths_to_add: List[str],
                                commit_message: str,
                                remote_name: Optional[str] = None,
                                branch_name: Optional[str] = None) -> bool:
        """
        Adds specified paths, commits, and pushes changes in the configured Git repository.

        Args:
            paths_to_add (List[str]): List of absolute paths to files/directories to add.
            commit_message (str): The commit message.
            remote_name (str, optional): The name of the remote repository. Overrides instance default.
            branch_name (str, optional): The name of the branch to push. Overrides instance default. If None, uses the instance's default_branch_name if set, otherwise pushes the current branch (HEAD).

        Returns:
            bool: True if all Git operations (add, commit, push) were successful, False otherwise.
        """
        # Initialization ensures self.repository_path is a valid Git repository and Git is executable.

        relative_paths_to_add = []
        for p_abs in paths_to_add:
            try:
                rel_path = os.path.relpath(p_abs, self.repository_path)
                relative_paths_to_add.append(rel_path)
            except ValueError:
                self.logger.error(f"Path {p_abs} cannot be made relative to repository {self.repository_path}. Skipping add for this path.")
        
        if not relative_paths_to_add:
            self.logger.warning("No valid relative paths provided to add to Git. Skipping Git operations.")
            return True

        if not self._run_git_command(['git', 'add'] + relative_paths_to_add):
            return False
        
        status_output = subprocess.run(['git', 'status', '--porcelain'], cwd=self.repository_path, capture_output=True, text=True, encoding='utf-8')
        if not status_output.stdout.strip():
            self.logger.info("No changes staged for commit. Skipping commit and push.")
            return True

        if not self._run_git_command(['git', 'commit', '-m', commit_message]):
            return False

        current_remote = remote_name if remote_name is not None else self.default_remote_name
        current_branch = branch_name # Can be None, which means use self.default_branch_name or HEAD

        push_command_parts = ['git', 'push', current_remote]
        if current_branch: # If a specific branch is provided for the method call
            push_command_parts.append(current_branch)
        elif self.default_branch_name: # Else, if a default branch is set for the instance
            push_command_parts.append(self.default_branch_name)
        else: # Else, push HEAD (current branch)
            push_command_parts.append('HEAD')

        if not self._run_git_command(push_command_parts):
            return False

        self.logger.info(f"Successfully added, committed, and pushed changes for: {commit_message} to {current_remote}/{push_command_parts[-1]}")
        return True