import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[PROC] Usage: python mapping_processor.py <input_a_parquet> <input_b_parquet> <key_a> <key_b>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_mapping.parquet"
    run = lambda input_a, input_b, key_a, key_b: (
        print(f"[PROC] Mapping started for: {input_a} and {input_b}") or
        (lambda df_a:
            (lambda df_b:
                (lambda mapped:
                    print(f"[PROC] Loaded shapes: A={df_a.shape}, B={df_b.shape}") or
                    print(f"[PROC] Mapped DataFrame shape: {mapped.shape}") or
                    mapped.write_parquet(get_output_filename(input_a)) or
                    print(f"[PROC] Mapping finished. Output: {get_output_filename(input_a)}")
                )(df_a.join(df_b, left_on=key_a, right_on=key_b, how="inner") if df_a.height > 0 and df_b.height > 0 else pl.DataFrame([]))
            )(pl.read_parquet(input_b))
        )(pl.read_parquet(input_a))
    )
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        else:
            input_a, input_b = args[1], args[2]
            key_a, key_b = args[3], args[4]
            run(input_a, input_b, key_a, key_b)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)