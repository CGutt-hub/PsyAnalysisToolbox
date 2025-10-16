import polars as pl, sys, os, functools
if __name__ == "__main__":
    usage = lambda: print("[PROC] Usage: python merging_processor.py <key1> <key2> ... <input1.parquet> <input2.parquet> ...") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_merged.parquet"
    run = lambda keys, input_files: (
        print(f"[PROC] Merging {len(input_files)} modalities on keys: {keys}") or
        (lambda dfs:
            (lambda merged:
                print(f"[PROC] Loaded shapes: {[df.shape for df in dfs]}") or
                print(f"[PROC] Merged DataFrame shape: {merged.shape}") or
                merged.write_parquet(get_output_filename(input_files[0])) or
                print(f"[PROC] Merge finished. Output: {get_output_filename(input_files[0])}")
            )(
                functools.reduce(
                    lambda acc, df: acc.join(df, on=keys, how='inner', suffix=f'_mod'),
                    dfs[1:],
                    dfs[0] if dfs else pl.DataFrame([])
                )
            )
        )([pl.read_parquet(f) for f in input_files])
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            keys = [k for k in args[1:] if not k.endswith('.parquet')]
            input_files = [f for f in args[1:] if f.endswith('.parquet')]
            run(keys, input_files)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)
