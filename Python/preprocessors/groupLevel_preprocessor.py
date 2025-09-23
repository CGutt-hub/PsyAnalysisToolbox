import polars as pl, sys, os, glob
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python groupLevel_preprocessor.py <input_dir> <output_parquet>") or sys.exit(1)
    # Lambda: find all participant PDFs and attached Parquet files in input_dir
    find_parquets = lambda input_dir: [f for pdf in glob.glob(os.path.join(input_dir, "*.pdf")) for f in glob.glob(os.path.splitext(pdf)[0] + "*.parquet")]
    # Lambda: read and concatenate all Parquet files into a group-level DataFrame
    aggregate_parquets = lambda parquet_files: pl.concat([pl.read_parquet(f) for f in parquet_files]) if parquet_files else pl.DataFrame([])
    # Lambda: clean group-level DataFrame (drop NaNs)
    clean_group_df = lambda df: df.drop_nulls() if df.height > 0 else df
    # Lambda: write group-level DataFrame to Parquet
    write_group_parquet = lambda df, output_parquet: df.write_parquet(output_parquet)
    # Lambda: main group-level preprocessing logic
    run = lambda input_dir, output_parquet: (
        print(f"[Nextflow] Group-level preprocessing started for: {input_dir}") or (
            # Lambda: find all Parquet files attached to participant PDFs
            (lambda parquet_files: (
                print(f"[Nextflow] Found {len(parquet_files)} Parquet files for aggregation."),
                # Lambda: aggregate all Parquet files
                (lambda group_df: (
                    print(f"[Nextflow] Aggregated group-level DataFrame shape: {group_df.shape}"),
                    # Lambda: clean group-level DataFrame
                    (lambda clean_df: (
                        print(f"[Nextflow] Cleaned group-level DataFrame shape: {clean_df.shape}"),
                        # Lambda: write output Parquet
                        write_group_parquet(clean_df, output_parquet),
                        print(f"[Nextflow] Group-level preprocessing finished. Output: {output_parquet}")
                    ))(clean_group_df(group_df))
                ))(aggregate_parquets(parquet_files))
            ))(find_parquets(input_dir))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_dir, output_parquet = args[1], args[2]
            run(input_dir, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Group-level preprocessing errored. Error: {e}")
        sys.exit(1)