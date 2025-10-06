import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python csv_reader.py <input_csv>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_csv.parquet"
    run = lambda input_csv: (
        print(f"[Nextflow] CSV Reader started for: {input_csv}") or
        (lambda df:
            print(f"[Nextflow] CSV file loaded: {input_csv}, shape: {df.shape}") or
            df.write_parquet(get_output_filename(input_csv)) or
            print(f"[Nextflow] Parquet file saved: {get_output_filename(input_csv)}")
        )(pl.read_csv(input_csv))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 2 else run(a[1]))(args)
    except Exception as e:
        print(f"[Nextflow] CSV Reader errored. Error: {e}")
        sys.exit(1)