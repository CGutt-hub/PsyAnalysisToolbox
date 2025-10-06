import sys, subprocess, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("[Nextflow] Usage: python git_sync.py <repo_path>") or sys.exit(1)
    # Lambda: main git sync logic, lean and nested
    run = lambda repo_path: (
        print(f"[Nextflow] Git Sync started for: {repo_path}") or
            (lambda:
                os.system(f"git -C {repo_path} pull") or
                print(f"[Nextflow] Git pull completed for: {repo_path}") or
                os.system(f"git -C {repo_path} push") or
                print(f"[Nextflow] Git push completed for: {repo_path}") or
                print(f"[Nextflow] Git sync finished.")
            )()
        )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 2 else run(a[1]))(args)
    except Exception as e:
        print(f"[Nextflow] GitSync: Error: {e}")
        sys.exit(1)