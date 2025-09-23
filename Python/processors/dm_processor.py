import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python dm_processor.py <events_parquet> <output_parquet>") or sys.exit(1)
    # Lambda: read events Parquet file
    read_parquet = lambda f: pl.read_parquet(f)
    # Lambda: build design matrix from events (generic columns: condition, onset, duration, amplitude)
    build_dm = lambda df: df.select([pl.col('condition'), pl.col('onset'), pl.col('duration'), pl.col('amplitude')]) if all(c in df.columns for c in ['condition','onset','duration','amplitude']) else pl.DataFrame([])
    # Lambda: write design matrix to Parquet
    write_parquet = lambda df, output_parquet: df.write_parquet(output_parquet)
    # Lambda: main DM processing logic
    run = lambda events_parquet, output_parquet: (
        print(f"[Nextflow] Design matrix processing started for: {events_parquet}") or (
            # Lambda: read events
            (lambda df: (
                print(f"[Nextflow] Loaded events DataFrame shape: {df.shape}"),
                # Lambda: build design matrix
                (lambda dm: (
                    print(f"[Nextflow] Built design matrix shape: {dm.shape}"),
                    # Lambda: write output Parquet
                    write_parquet(dm, output_parquet),
                    print(f"[Nextflow] Design matrix processing finished. Output: {output_parquet}")
                ))(build_dm(df))
            ))(read_parquet(events_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            events_parquet, output_parquet = args[1], args[2]
            run(events_parquet, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Design matrix processing errored. Error: {e}")
        sys.exit(1)