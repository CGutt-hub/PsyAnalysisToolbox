import polars as pl, sys, re, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python questionnaire_analyzer.py <input_parquet> <answer_key_pattern> <tick_pattern>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_quest.parquet"
    run = lambda input_parquet, answer_key_pattern, tick_pattern: (
        print(f"[Nextflow] Questionnaire analysis started for: {input_parquet}") or (
            (lambda df:
                print(f"[Nextflow] Loaded questionnaire DataFrame shape: {df.shape}") or (
                    (lambda keys:
                        print(f"[Nextflow] Found answer key columns: {keys}") or (
                            (lambda ticks:
                                print(f"[Nextflow] Found tick columns: {ticks}") or (
                                    (lambda extracted:
                                        print(f"[Nextflow] Extracted data shape: {extracted.shape}") or (
                                            extracted.write_parquet(get_output_filename(input_parquet)),
                                            print(f"[Nextflow] Questionnaire analysis finished. Output: {get_output_filename(input_parquet)}")
                                        )
                                    )(df.select([pl.col(k) for k in keys + ticks if k in df.columns]) if keys or ticks else pl.DataFrame([]))
                                )
                            )([col for col in df.columns if re.search(tick_pattern, col)])
                        )
                    )([col for col in df.columns if re.search(answer_key_pattern, col)])
                )
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            input_parquet = args[1]
            answer_key_pattern = args[2]
            tick_pattern = args[3]
            run(input_parquet, answer_key_pattern, tick_pattern)
    except Exception as e:
        print(f"[Nextflow] Questionnaire analysis errored. Error: {e}")
        sys.exit(1)