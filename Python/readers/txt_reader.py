import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[READER] Usage: python txt_reader.py <input_txt> <encoding>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_txt.parquet"
    run = lambda input_txt, encoding: (
        print(f"[READER] Started for: {input_txt}") or
        (lambda lines:
            print(f"[READER] TXT file loaded: {input_txt}, lines: {len(lines)}") or
            (lambda df:
                print(f"[READER] DataFrame created with shape: {df.shape}") or
                df.write_parquet(get_output_filename(input_txt)) or
                print(f"[READER] Parquet file saved: {get_output_filename(input_txt)}")
            )(pl.DataFrame({
                input_txt.split('/')[-1]: lines  # Use filename as column name
            }))
        )(
            # Read file as plain text lines, preserving all content including tabs
            open(input_txt, 'r', encoding=encoding).read().split('\n')
        )
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
        print(f"[READER] Error: {e}")
        sys.exit(1)
