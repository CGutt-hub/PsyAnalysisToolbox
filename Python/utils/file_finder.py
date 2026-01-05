import os, sys, fnmatch, shutil, polars as pl

def find_in_subdir(signal_file, pattern):
    # Read folder path from signal file
    try:
        df = pl.read_parquet(signal_file)
        if 'folder_path' in df.columns:
            folder_path = df['folder_path'][0]
            print(f"[FINDER] Using folder from signal: {folder_path}", file=sys.stderr)
        else:
            # Fallback: derive from signal file name
            real_path = os.path.realpath(signal_file)
            signal_dir = os.path.dirname(real_path)
            signal_name = os.path.splitext(os.path.basename(real_path))[0]
            folder_path = os.path.join(signal_dir, signal_name)
            print(f"[FINDER] No folder_path in signal, using fallback: {folder_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading signal file: {e}", file=sys.stderr)
        return []
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}", file=sys.stderr)
        return []
    
    matches = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if fnmatch.fnmatch(f, pattern)]
    if not matches:
        print(f"Error: No files matching '{pattern}' in {folder_path}", file=sys.stderr)
    return matches

def copy_and_output(matches):
    if not matches: return 1
    for f in matches: shutil.copy2(f, b := os.path.basename(f)); print(b)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3: print("Usage: python file_finder.py <signal_file> <pattern>", file=sys.stderr); sys.exit(1)
    matches = find_in_subdir(sys.argv[1], sys.argv[2])
    sys.exit(copy_and_output(matches))


