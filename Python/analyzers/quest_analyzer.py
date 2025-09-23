import polars as pl, sys, re
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python questionnaire_analyzer.py <input_parquet> <answer_key_pattern> <tick_pattern> <output_parquet>") or sys.exit(1)
    # Lambda: read preprocessed questionnaire Parquet
    read_parquet = lambda f: pl.read_parquet(f)
    # Lambda: find answer key columns by regex pattern
    find_keys = lambda df, pattern: [col for col in df.columns if re.search(pattern, col)]
    # Lambda: find tick columns by regex pattern
    find_ticks = lambda df, pattern: [col for col in df.columns if re.search(pattern, col)]
    # Lambda: extract answer/tick data
    extract_data = lambda df, keys, ticks: df.select([pl.col(k) for k in keys + ticks if k in df.columns]) if keys or ticks else pl.DataFrame([])
    # Lambda: write extracted data to Parquet
    write_parquet = lambda df, output_parquet: df.write_parquet(output_parquet)
    # Lambda: main questionnaire analysis logic
    run = lambda input_parquet, answer_key_pattern, tick_pattern, output_parquet: (
        print(f"[Nextflow] Questionnaire analysis started for: {input_parquet}") or (
            (lambda df:
                print(f"[Nextflow] Loaded questionnaire DataFrame shape: {df.shape}") or (
                    (lambda keys:
                        print(f"[Nextflow] Found answer key columns: {keys}") or (
                            (lambda ticks:
                                print(f"[Nextflow] Found tick columns: {ticks}") or (
                                    (lambda extracted:
                                        print(f"[Nextflow] Extracted data shape: {extracted.shape}") or (
                                            write_parquet(extracted, output_parquet),
                                            print(f"[Nextflow] Questionnaire analysis finished. Output: {output_parquet}")
                                        )
                                    )(extract_data(df, keys, ticks))
                                )
                            )(find_ticks(df, tick_pattern))
                        )
                    )(find_keys(df, answer_key_pattern))
                )
            )(read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        else:
            input_parquet = args[1]
            answer_key_pattern = args[2]
            tick_pattern = args[3]
            output_parquet = args[4]
            run(input_parquet, answer_key_pattern, tick_pattern, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Questionnaire analysis errored. Error: {e}")
        sys.exit(1)