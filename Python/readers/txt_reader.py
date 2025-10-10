import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python txt_reader.py <input_txt> <encoding>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_txt.parquet"
    run = lambda input_txt, encoding: (
        print(f"[Nextflow] TXT Reader started for: {input_txt}") or
        (lambda df:
            print(f"[Nextflow] TXT file loaded: {input_txt}, shape: {df.shape}") or
            df.write_parquet(get_output_filename(input_txt)) or
            print(f"[Nextflow] Parquet file saved: {get_output_filename(input_txt)}")
        )(pl.read_csv(input_txt, separator='\t', encoding=encoding, truncate_ragged_lines=True, ignore_errors=True))
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_txt = args[1]
            encoding = args[2].strip()
            run(input_txt, encoding)
    except Exception as e:
        print(f"[Nextflow] TXT Reader errored. Error: {e}")
        sys.exit(1)
