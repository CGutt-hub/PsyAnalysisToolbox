import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python quest_preprocessor.py <input_parquet> <encoding> <output_dir>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_quest_preproc.parquet"
    
    run = lambda input_parquet, encoding, output_dir: (
        print(f"[Nextflow] Questionnaire preprocessor started for: {input_parquet}") or
        (lambda df:
            print(f"[Nextflow] Read DataFrame with shape: {df.shape}") or
            print(f"[Nextflow] DataFrame columns: {df.columns}") or
            (lambda text_lines:
                print(f"[Nextflow] Extracted {len(text_lines)} text lines") or
                (lambda questionnaire_data:
                    print(f"[Nextflow] Extracted {len(questionnaire_data)} questionnaire entries") or
                    (lambda result_df:
                        print(f"[Nextflow] Created DataFrame with shape: {result_df.shape}") or
                        print(f"[Nextflow] Sample keys: {result_df['key'].head(10).to_list() if result_df.shape[0] > 0 else 'None'}") or
                        result_df.write_parquet(get_output_filename(input_parquet)) or
                        print(f"[Nextflow] Questionnaire preprocessing finished. Output: {get_output_filename(input_parquet)}")
                    )(pl.DataFrame(questionnaire_data) if questionnaire_data else pl.DataFrame({"key": [], "value": []}))
                )([
                    {"key": parts[0].strip(), "value": parts[1].strip()}
                    for line in text_lines
                    if line is not None and isinstance(line, str) and ':' in line 
                    and not line.strip().startswith('***') and not line.strip().endswith('***')
                    for parts in [line.split(':', 1)]
                    if len(parts) == 2 and parts[0] is not None and parts[1] is not None 
                    and parts[0].strip() and parts[1].strip()
                    # Enhanced filtering for questionnaire-relevant data
                    and (
                        # Include questionnaire names
                        any(quest in parts[0].lower() for quest in ['panas', 'bis', 'bas', 'sam', 'ea11', 'be7']) or
                        # Include response patterns
                        any(pattern in parts[0].lower() for pattern in ['response', 'answer', 'rating', 'score', 'resp']) or
                        # Include timing and behavioral data  
                        any(pattern in parts[0].lower() for pattern in ['rt', 'acc', 'onset', 'duration', 'trigger']) or
                        # Include trial and stimulus information
                        any(pattern in parts[0].lower() for pattern in ['trial', 'stimulus', 'item', 'question']) or
                        # Standard key-value pairs (keep existing functionality)
                        True
                    )
                ])
            )(
                # Extract text lines from the polars DataFrame, filtering out None values
                [str(item) for item in df.to_series(0).to_list() if item is not None] if df.shape[0] > 0 else []
            )
        )(pl.read_parquet(input_parquet))
    )
    
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            input_parquet = args[1]
            encoding = args[2]  # Not used anymore but kept for compatibility
            output_dir = args[3]
            run(input_parquet, encoding, output_dir)
    except Exception as e:
        print(f"[Nextflow] Questionnaire preprocessing errored. Error: {e}")
        sys.exit(1)