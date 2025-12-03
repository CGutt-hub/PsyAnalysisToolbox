import polars as pl, sys, os

# usage lambda
usage = lambda: (print('[PROC] Usage: python extracting_processor.py <input_parquet> <selector1> [selector2 ...]') or sys.exit(1))

get_output_filename = lambda base: f"{base}_extr.parquet"



# Use lambdas for concise logic, named functions for multi-step logic
def resolver(s, dcols):
    # Allow both numeric and name-based selectors
    if s is None or s == '':
        return []
    s = s.strip()
    # range
    if ':' in s:
        parts = s.split(':')
        start = 0 if parts[0].strip() == '' else (int(parts[0].strip()) - 1 if not parts[0].strip().startswith('-') else len(dcols) + int(parts[0].strip()))
        end = len(dcols) - 1 if (len(parts) <= 1 or parts[1].strip() == '') else (int(parts[1].strip()) - 1 if not parts[1].strip().startswith('-') else len(dcols) + int(parts[1].strip()))
        return dcols[start:end+1] if start <= end else []
    if s.lstrip('-').isdigit():
        idx = int(s)
        if 1 <= idx <= len(dcols):
            return [dcols[idx - 1]]
        if idx < 0 and abs(idx) <= len(dcols):
            return [dcols[len(dcols) + idx]]
        return []
    name = s.lower()
    return ([c for c in dcols if c.lower() == name] or
            [c for c in dcols if name in c.lower()] or
            [c for c in dcols if c.lower().startswith(name)] or [])

def write_outputs(df, selectors, base, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    # determine data columns (exclude time if present at first position)
    dcols = (df.columns[1:] if df.columns and df.columns[0].lower() == 'time' else df.columns)
    writes = 0
    unresolved = []
    for idx, sel in enumerate(selectors):
        try:
            sel_cols = resolver(sel, dcols)
        except ValueError as e:
            raise
        if not sel_cols:
            unresolved.append((idx + 1, sel))
            continue
        # always include time column if present
        if 'time' in df.columns:
            out_df = df.select(['time'] + sel_cols)
        else:
            # should not happen because we validate earlier, but keep safe
            out_df = df.select(sel_cols)
        out_path = os.path.join(out_folder, f"{base}_extr{idx+1}.parquet")
        out_df.write_parquet(out_path)
        print(f"[PROC] Wrote {os.path.basename(out_path)} columns={out_df.columns}")
        writes += 1
    if unresolved:
        avail = ','.join(dcols)
        msg = f"Selectors that resolved to no columns: {unresolved}. Available columns (excluding time): {avail}"
        raise ValueError(msg)
    return writes

write_signal = lambda input_parquet, base, out_folder, writes_count: (
    pl.DataFrame({'signal':[1], 'source':[os.path.basename(input_parquet)], 'streams':[writes_count]})
        .write_parquet(os.path.join(out_folder, get_output_filename(base))) or
    print(f"[PROC] Extraction finished. Wrote signal {os.path.join(out_folder, get_output_filename(base))} with {writes_count} streams")
)

def run(input_parquet, selectors):
    print(f"[PROC] Extracting started for: {input_parquet} selectors={selectors}")
    df = pl.read_parquet(input_parquet)
    # require a time column in per-stream files
    if 'time' not in [c.lower() for c in df.columns]:
        base = os.path.splitext(os.path.basename(input_parquet))[0]
        raise ValueError(f"Input file '{input_parquet}' does not contain a 'time' column and looks like a signalling/metadata file.\nUse the per-stream parquet located in the '{base}_xdf' folder (e.g. '{base}_xdf/{base}_xdf4.parquet').")
    base = os.path.splitext(os.path.basename(input_parquet))[0]
    # Always create the extr folder in the workspace root (cwd)
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_extr")
    writes_count = write_outputs(df, selectors, base, out_folder)
    # Write the extr signalling file directly in the workspace root
    write_signal(input_parquet, base, workspace_root, writes_count)

if __name__ == '__main__':
    # simple CLI: python extracting_processor.py <input_parquet> <selector1> [selector2 ...]
    if len(sys.argv) < 3:
        usage()
    input_parquet = sys.argv[1]
    selectors = sys.argv[2:]
    try:
        run(input_parquet, selectors)
    except Exception as e:
        print(f"[PROC][ERROR] {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            selectors = args[2:] if len(args) > 2 else []
            run(input_parquet, selectors)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)
