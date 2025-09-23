import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python mapping_processor.py <input_a_parquet> <input_b_parquet> <key_a> <key_b> <output_parquet>") or sys.exit(1)
    # Lambda: read Parquet files
    read_parquet = lambda f: pl.read_parquet(f)
    # Lambda: main mapping logic (generic join by user-specified keys)
    run = lambda input_a, input_b, key_a, key_b, output_parquet: (
        print(f"[Nextflow] Mapping started for: {input_a} and {input_b}") or (
            # Lambda: read both datasets
            (lambda df_a, df_b: (
                print(f"[Nextflow] Loaded shapes: A={df_a.shape}, B={df_b.shape}"),
                # Lambda: join A and B on specified keys
                (lambda mapped: (
                    print(f"[Nextflow] Mapped DataFrame shape: {mapped.shape}"),
                    # Lambda: write mapped DataFrame to Parquet
                    mapped.write_parquet(output_parquet),
                    print(f"[Nextflow] Mapping finished. Output: {output_parquet}")
                ))(df_a.join(df_b, left_on=key_a, right_on=key_b, how="inner") if df_a.height > 0 and df_b.height > 0 else pl.DataFrame([]))
            ))(read_parquet(input_a), read_parquet(input_b))
        )
    )
    try:
        args = sys.argv
        if len(args) < 6:
            usage()
        else:
            input_a, input_b = args[1], args[2]
            key_a, key_b = args[3], args[4]
            output_parquet = args[5]
            run(input_a, input_b, key_a, key_b, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Mapping errored. Error: {e}")
        sys.exit(1)