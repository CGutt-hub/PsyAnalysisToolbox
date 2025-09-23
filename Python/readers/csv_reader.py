import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python csv_reader.py <input_csv> <output_parquet>") or sys.exit(1)
    # Lambda: main CSV reading logic, maximally nested
    run = lambda input_csv, output_parquet: (
        # Lambda: print start message
        print(f"[Nextflow] CSV reading started for: {input_csv}") or (
            # Lambda: read CSV and process DataFrame
            (lambda df:
                # Lambda: print DataFrame shape
                print(f"[Nextflow] Loaded CSV DataFrame shape: {df.shape}") or (
                    # Lambda: write DataFrame to Parquet
                    (lambda _: df.write_parquet(output_parquet))(df) or (
                        # Lambda: print finished message
                        print(f"[Nextflow] CSV reading finished. Output: {output_parquet}")
                    )
                )
            )(pl.read_csv(input_csv))
        )
    )
    try:
        args = sys.argv
        # Lambda: check argument count and run main logic
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] CSV reading errored. Error: {e}")
        sys.exit(1)