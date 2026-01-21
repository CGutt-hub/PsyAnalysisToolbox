import polars as pl, sys, os, functools

def merge_files(keys: list[str], files: list[str]) -> str:
    for f in files:
        if not os.path.exists(f): print(f"[merging] Error: File not found: {f}"); sys.exit(1)
    if not keys: print("[merging] Error: No join keys specified"); sys.exit(1)
    if len(files) < 2: print("[merging] Error: Need at least 2 files to merge"); sys.exit(1)
    print(f"[merging] Merging {len(files)} files on keys: {keys}")
    dfs = [pl.read_parquet(f) for f in files]
    for i, df in enumerate(dfs):
        missing = [k for k in keys if k not in df.columns]
        if missing: print(f"[merging] Error: Keys {missing} not in {files[i]}"); sys.exit(1)
    merged = functools.reduce(lambda acc, df: acc.join(df, on=keys, how='inner', suffix='_mod'), dfs[1:], dfs[0])
    out = f"{os.path.splitext(os.path.basename(files[0]))[0]}_merged.parquet"
    merged.write_parquet(out)
    print(f"[merging] Output: {out} ({merged.shape})")
    return out

if __name__ == '__main__': (lambda a: merge_files([k for k in a[1:] if not k.endswith('.parquet')], [f for f in a[1:] if f.endswith('.parquet')]) if len(a) >= 3 else (print('[merging] Merge multiple parquet files on shared key columns.\nUsage: merging_processor.py <key1> [key2...] <f1.parquet> <f2.parquet> ...'), sys.exit(1)))(sys.argv)
