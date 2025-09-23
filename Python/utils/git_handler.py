import sys, subprocess, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python git_handler.py <repo_path> <commit_msg> <file1> [<file2> ...]") or sys.exit(1)
    # Lambda: main git sync logic, with nested lambdas and comments for each step
    run = lambda repo, msg, files, remote='origin': (
        # Lambda: print start message
        print(f"[Nextflow] GitHandler: Adding files to repo {repo}") or (
            # Lambda: add each file to git staging
            (lambda _: [
                # Lambda: git add for each file
                (lambda __: subprocess.run(['git', 'add', f], cwd=repo, check=True))(f)
                for f in files
            ])(files) or (
                # Lambda: print commit message
                print(f"[Nextflow] GitHandler: Committing with message: {msg}") or (
                    # Lambda: git commit
                    (lambda _: subprocess.run(['git', 'commit', '-m', msg], cwd=repo, check=True))(msg) or (
                        # Lambda: print push message
                        print(f"[Nextflow] GitHandler: Pushing to remote: {remote}") or (
                            # Lambda: git push
                            (lambda _: subprocess.run(['git', 'push', remote], cwd=repo, check=True))(remote) or (
                                # Lambda: print sync complete
                                print("[Nextflow] GitHandler: Sync complete.")
                            )
                        )
                    )
                )
            )
        )
    )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 4 else run(a[1], a[2], a[3:]))(args)
    except Exception as e:
        print(f"[Nextflow] GitHandler: Error: {e}")
        sys.exit(1)