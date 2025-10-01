import sys, subprocess, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python git_sync.py <plot_file> <participant_id>") or sys.exit(1)
    # Lambda: main git sync logic, lean and nested
    run = lambda plot_file, participant_id, repo='.', remote='origin': (
        # Lambda: print start message
        print(f"[Nextflow] GitSync: Adding plot file {plot_file} for participant {participant_id}") or (
            # Lambda: git add plot file
            (lambda _: subprocess.run(['git', 'add', plot_file], cwd=repo, check=True))(plot_file) or (
                # Lambda: print commit message
                print(f"[Nextflow] GitSync: Committing with message: Auto sync for {participant_id}") or (
                    # Lambda: git commit
                    (lambda _: subprocess.run(['git', 'commit', '-m', f"Auto sync for {participant_id}"], cwd=repo, check=True))(participant_id) or (
                        # Lambda: print push message
                        print(f"[Nextflow] GitSync: Pushing to remote: {remote}") or (
                            # Lambda: git push
                            (lambda _: subprocess.run(['git', 'push', remote], cwd=repo, check=True))(remote) or (
                                # Lambda: print sync complete
                                print("[Nextflow] GitSync: Sync complete.")
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
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] GitSync: Error: {e}")
        sys.exit(1)