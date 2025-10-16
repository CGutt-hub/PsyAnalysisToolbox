import sys, subprocess, os
if __name__ == "__main__":
    usage = lambda: print("[UTIL] Usage: python git_sync.py <repo_path>") or sys.exit(1)
    run = lambda repo_path: (
        print(f"[UTIL] Git Sync started for: {repo_path}") or
            (lambda:
                os.system(f"git -C {repo_path} pull") or
                print(f"[UTIL] Git pull completed for: {repo_path}") or
                os.system(f"git -C {repo_path} push") or
                print(f"[UTIL] Git push completed for: {repo_path}") or
                print(f"[UTIL] Git sync finished.")
            )()
        )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 2 else run(a[1]))(args)
    except Exception as e:
        print(f"[UTIL] Error: {e}")
        sys.exit(1)