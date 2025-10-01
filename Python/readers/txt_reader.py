import polars as pl, sys
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python txt_reader.py <input_txt> <output_parquet>") or sys.exit(1)
    run = lambda input_txt, output_parquet: (
        print(f"[Nextflow] TXT Reader started for: {input_txt}"),
        (lambda df: (
            print(f"[Nextflow] TXT file loaded: {input_txt}, shape: {df.shape}"),
            (lambda _: (
                print(f"[Nextflow] Writing DataFrame to Parquet: {output_parquet}"),
                df.write_parquet(output_parquet),
                print(f"[Nextflow] Parquet file saved: {output_parquet}")
            ))(df)
        ))(pl.read_csv(input_txt, separator='\t'))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2]))(args)
    except Exception as e:
        print(f"[Nextflow] TXT Reader errored. Error: {e}")
        sys.exit(1)
