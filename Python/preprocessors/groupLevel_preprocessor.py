import polars as pl, sys, os, glob
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("[PREPROC] Usage: python groupLevel_preprocessor.py <input_dir>") or sys.exit(1)
    get_output_filename = lambda input_dir: f"{os.path.basename(os.path.normpath(input_dir))}_groupLevel.parquet"
    run = lambda input_dir: (
        print(f"[PREPROC] Group-level preprocessing started for: {input_dir}") or
        (lambda parquet_files: (
            print(f"[PREPROC] Found {len(parquet_files)} Parquet files for aggregation."),
            (lambda group_df: (
                print(f"[PREPROC] Aggregated group-level DataFrame shape: {group_df.shape}"),
                (lambda clean_df: (
                    print(f"[PREPROC] Cleaned group-level DataFrame shape: {clean_df.shape}"),
                    clean_df.write_parquet(get_output_filename(input_dir)),
                    print(f"[PREPROC] Group-level preprocessing finished. Output: {get_output_filename(input_dir)}")
                ))(group_df.drop_nulls() if group_df.height > 0 else group_df)
            ))(pl.concat([pl.read_parquet(f) for f in parquet_files]) if parquet_files else pl.DataFrame([]))
        ))([f for pdf in glob.glob(os.path.join(input_dir, "*.pdf")) for f in glob.glob(os.path.splitext(pdf)[0] + "*.parquet")])
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_dir = args[1]
            run(input_dir)
    except Exception as e:
        print(f"[PREPROC] Error: {e}")
        sys.exit(1)