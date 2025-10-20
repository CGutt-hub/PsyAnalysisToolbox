import sys, polars as pl, os, tempfile, shutil
if __name__ == "__main__":
    usage = lambda: print("[PROC] Usage: python extracting_processor.py <input_parquet> <col1> [col2] [col3] ... [colN]") or sys.exit(1)
    get_column_selection = lambda df, spec: (
        df.select(df.columns[int(spec.split(":")[0]) if spec.split(":")[0] else 0 :
                           (len(df.columns) + int(spec.split(":")[1])) if spec.split(":")[1] and int(spec.split(":")[1]) < 0
                           else (int(spec.split(":")[1]) if spec.split(":")[1] else len(df.columns))]) if ":" in spec
        else df.select([spec]) if spec in df.columns
        else None
    )
    run = lambda input_path, columns: (
        (lambda df:
            (lambda base:
                (lambda temp_dir:
                    print(f"[PROC] Using temp dir: {temp_dir}") or
                    (
                        [
                            (lambda idx, col_spec:
                                (lambda selected_df:
                                    selected_df.write_parquet(os.path.join(temp_dir, f"{base}_extr{idx+1}.parquet")) if selected_df is not None
                                    else print(f"[PROC] Warning: Column spec '{col_spec}' not found or invalid")
                                )(get_column_selection(df, col_spec))
                            )(idx, col_spec)
                            for idx, col_spec in enumerate(columns)
                        ]
                    ) or (
                        # Move all files from temp dir to current dir atomically
                        [shutil.move(os.path.join(temp_dir, f), os.path.join(".", f)) for f in os.listdir(temp_dir) if f.endswith('.parquet')] or
                        (lambda _: (shutil.rmtree(temp_dir), True))(None) or
                        # write a tiny canonical signal parquet for dispatcher detection
                        (lambda sig: (pl.DataFrame({'signal':[1], 'base':[base]}).write_parquet(sig), print(f"[PROC] Channel extraction completed. Base: {base}, Extracts: {len(columns)}")))(f"{base}_extr.parquet")
                    )
                )(tempfile.mkdtemp(prefix="extract_temp_", dir="."))
            )(os.path.splitext(os.path.basename(input_path))[0])
        )(pl.read_parquet(input_path))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 3 else run(a[1], a[2:]))(args)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)
        sys.exit(1)