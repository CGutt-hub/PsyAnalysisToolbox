import polars as pl, sys
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python csv_reader.py <input_csv> <output_parquet>") or sys.exit(1)
    run = lambda input_csv, output_parquet: (
        print(f"[Nextflow] CSV Reader started for: {input_csv}"),
        (lambda df: (
            print(f"[Nextflow] CSV file loaded: {input_csv}, shape: {df.shape}"),
            (lambda _: (
                print(f"[Nextflow] Writing DataFrame to Parquet: {output_parquet}"),
                df.write_parquet(output_parquet),
                print(f"[Nextflow] Parquet file saved: {output_parquet}")
            ))(df)
        ))(pl.read_csv(input_csv))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] CSV Reader errored. Error: {e}")
        sys.exit(1)