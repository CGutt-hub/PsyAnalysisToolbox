import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python txt_reader.py <input_txt>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_txt.parquet"
    run = lambda input_txt: (
        print(f"[Nextflow] TXT Reader started for: {input_txt}") or
        (lambda df:
            print(f"[Nextflow] TXT file loaded: {input_txt}, shape: {df.shape}") or
            df.write_parquet(get_output_filename(input_txt)) or
            print(f"[Nextflow] Parquet file saved: {get_output_filename(input_txt)}")
        )(pl.read_csv(input_txt, separator='\t'))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 2 else run(a[1]))(args)
    except Exception as e:
        print(f"[Nextflow] TXT Reader errored. Error: {e}")
        sys.exit(1)
