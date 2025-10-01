import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python merging_processor.py <output_parquet> <key1> <key2> ... <input1.parquet> <input2.parquet> ...") or sys.exit(1)
    # Lambda: parse arguments into keys and input files
    parse_args = lambda args: (
        # First argument is output, next N are keys, rest are input files
        (args[1], [k for k in args[2:] if not k.endswith('.parquet')], [f for f in args[2:] if f.endswith('.parquet')])
    )
    # Lambda: read all input Parquet files
    read_all = lambda files: [pl.read_parquet(f) for f in files]
    # Lambda: merge all DataFrames on shared keys
    merge_all = lambda dfs, keys: (
        # Iteratively join all DataFrames on keys
        (lambda merged: (
            [merged := merged.join(df, on=keys, how='inner', suffix=f'_mod{i}') for i, df in enumerate(dfs[1:], 2)],
            merged
        )[1] if dfs else pl.DataFrame([])
        )(dfs[0])
    )
    # Lambda: write merged DataFrame to Parquet
    write_parquet = lambda df, output_parquet: df.write_parquet(output_parquet)
    # Lambda: main merging logic
    run = lambda output_parquet, keys, input_files: (
        print(f"[Nextflow] Merging {len(input_files)} modalities on keys: {keys}") or (
            (lambda dfs: (
                print(f"[Nextflow] Loaded shapes: {[df.shape for df in dfs]}") or (
                    (lambda merged_df: (
                        print(f"[Nextflow] Merged DataFrame shape: {merged_df.shape}") or (
                            write_parquet(merged_df, output_parquet),
                            print(f"[Nextflow] Merge finished. Output: {output_parquet}")
                        )
                    ))(merge_all(dfs, keys))
                )
            ))(read_all(input_files))
        )
    )
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        else:
            output_parquet, keys, input_files = parse_args(args)
            run(output_parquet, keys, input_files)
    except Exception as e:
        print(f"[Nextflow] merging_processor errored. Error: {e}")
        sys.exit(1)
