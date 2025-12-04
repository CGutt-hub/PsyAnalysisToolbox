#!/usr/bin/env python3
"""Pipeline Watcher - Auto-trigger Nextflow for new participant folders."""
import sys, time, subprocess, json
from pathlib import Path
from datetime import datetime

log = lambda m, l='INFO': print(f"[{datetime.now():%H:%M:%S}] [{l}] {m}")

def load_state(f): 
    return set(json.load(open(f)).get('processed', [])) if f.exists() else set()

def save_state(f, p): 
    json.dump({'processed': sorted(list(p)), 'last_updated': datetime.now().isoformat()}, open(f, 'w'), indent=2)

def is_complete(folder):
    """Check if participant folder has required .xdf and .txt files."""
    pid = folder.name
    return (folder / f"{pid}.xdf").exists() and (folder / f"{pid}.txt").exists()

def run_pipeline(pipeline, participant, dry_run=False):
    """Execute Nextflow pipeline for participant."""
    log(f"Processing: {participant.name}")
    if dry_run:
        log(f"[DRY RUN] Would run: nextflow run {pipeline} -resume")
        return True
    try:
        result = subprocess.run(
            ['nextflow', 'run', str(pipeline), '-resume'],
            cwd=pipeline.parent,
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.returncode == 0:
            log(f"Success: {participant.name}")
            return True
        else:
            log(f"Failed: {participant.name} (exit {result.returncode})", 'ERROR')
            log(result.stderr, 'ERROR')
            return False
    except subprocess.TimeoutExpired:
        log(f"Timeout: {participant.name}", 'ERROR')
        return False
    except Exception as e:
        log(f"Error: {participant.name} - {e}", 'ERROR')
        return False

def watch(watch_dir, pipeline, pattern='EV_*', interval=30, state_file=None, dry_run=False, verbose=False):
    """Main watch loop."""
    watch_dir = Path(watch_dir).resolve()
    pipeline = Path(pipeline).resolve()
    state_file = Path(state_file) if state_file else watch_dir / '.pipeline_watcher_state.json'
    
    if not watch_dir.exists():
        raise ValueError(f"Watch directory not found: {watch_dir}")
    if not pipeline.exists():
        raise ValueError(f"Pipeline not found: {pipeline}")
    
    processed = load_state(state_file)
    log(f"Watching: {watch_dir}")
    log(f"Pipeline: {pipeline}")
    log(f"Pattern: {pattern}, Interval: {interval}s, Dry-run: {dry_run}")
    log(f"Previously processed: {len(processed)} participants")
    
    try:
        while True:
            new = [f for f in watch_dir.glob(pattern) 
                   if f.is_dir() and f.name not in processed and is_complete(f)]
            
            if new:
                log(f"Found {len(new)} new participant(s): {[f.name for f in new]}")
                for folder in new:
                    if run_pipeline(pipeline, folder, dry_run):
                        processed.add(folder.name)
                        save_state(state_file, processed)
                        log(f"Marked as processed: {folder.name}")
                    else:
                        log(f"Will retry: {folder.name}", 'WARN')
            elif verbose:
                log(f"No new participants", 'DEBUG')
            
            time.sleep(interval)
    except KeyboardInterrupt:
        log("Stopped by user")
        sys.exit(0)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Watch for new participants and trigger Nextflow pipeline')
    p.add_argument('watch_dir', help='Directory to watch')
    p.add_argument('pipeline_nf', help='Nextflow pipeline file')
    p.add_argument('--pattern', default='EV_*', help='Folder pattern (default: EV_*)')
    p.add_argument('--interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    p.add_argument('--state-file', help='State file path (default: .pipeline_watcher_state.json in watch_dir)')
    p.add_argument('--dry-run', action='store_true', help='Test without executing')
    p.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = p.parse_args()
    
    try:
        watch(args.watch_dir, args.pipeline_nf, args.pattern, args.interval, 
              args.state_file, args.dry_run, args.verbose)
    except Exception as e:
        log(f"Fatal: {e}", 'ERROR')
        sys.exit(1)
