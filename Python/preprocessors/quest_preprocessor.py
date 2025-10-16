import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("[PREPROC] Usage: python quest_preprocessor.py <input_parquet> <encoding> <output_dir> <trial_markers> <trigger_markers> <procedure_markers> <condition_markers> <positive_patterns> <negative_patterns> <neutral_patterns> <aggregate_within_conditions>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_quest_preproc.parquet"
    
    run = lambda input_parquet, encoding, output_dir, trial_params: (
        print(f"[PREPROC] Questionnaire preprocessor started for: {input_parquet}") or
        print(f"[PREPROC] Using trial parameters: {trial_params}") or
        (lambda df:
            print(f"[PREPROC] Read DataFrame with shape: {df.shape}") or
            print(f"[PREPROC] DataFrame columns: {df.columns}") or
            (lambda text_lines:
                print(f"[PREPROC] Extracted {len(text_lines)} text lines") or
                (lambda questionnaire_data:
                    print(f"[PREPROC] Extracted {len(questionnaire_data)} questionnaire entries") or
                    (lambda result_df:
                        print(f"[PREPROC] Created DataFrame with shape: {result_df.shape}") or
                        print(f"[PREPROC] Sample keys: {result_df['key'].head(10).to_list() if result_df.shape[0] > 0 else 'None'}") or
                        (lambda final_df:
                            final_df.write_parquet(get_output_filename(input_parquet)) or
                            print(f"[PREPROC] Questionnaire preprocessing finished. Output: {get_output_filename(input_parquet)}")
                        )(
                            # Apply within-condition aggregation if requested
                            (lambda aggregated_data:
                                print(f"[PREPROC] Applied within-condition aggregation: {trial_params['aggregate_within_conditions']}") or
                                (aggregated_data if aggregated_data.shape[0] > 0 else result_df)
                            )(
                                # Within-condition aggregation using pure polars operations (no JSON)
                                (lambda grouped_df:
                                    grouped_df.group_by(['condition', 'key']).agg([
                                        pl.col('value').cast(pl.Float64, strict=False).mean().alias('value_mean'),
                                        pl.col('value').cast(pl.Float64, strict=False).std().alias('value_variance'),
                                        pl.col('value').cast(pl.Float64, strict=False).count().alias('trial_count'),
                                        pl.first('trial_number').alias('trial_number'),
                                        pl.first('stimulus_file').alias('stimulus_file'),
                                        pl.first('trigger').alias('trigger'),
                                        pl.first('procedure').alias('procedure')
                                    ]).with_columns([
                                        (pl.col('key') + pl.lit('_aggregated')).alias('key'),
                                        pl.col('value_mean').cast(pl.Utf8).alias('value'),
                                        pl.col('value_variance').alias('variance')
                                    ]).select(['key', 'value', 'variance', 'condition', 'trial_count', 'trial_number', 'stimulus_file', 'trigger', 'procedure'])
                                )(result_df.filter(pl.col('condition').is_not_null())) if trial_params['aggregate_within_conditions'] and 'condition' in (result_df.columns if result_df.shape[0] > 0 else []) else pl.DataFrame()
                            )
                        )
                    )(
                        # Create DataFrame with dynamic columns based on available data
                        pl.DataFrame(questionnaire_data) if questionnaire_data else pl.DataFrame({"key": [], "value": []})
                    )
                )(
                    # Generic questionnaire data extraction with optional trial segmentation
                    (lambda parsed_data:
                        print(f"[PREPROC] Parsed {len(parsed_data)} questionnaire entries") or
                        parsed_data
                    )(
                        # Parse text lines with optional trial context tracking
                        (lambda questionnaire_entries:
                            [entry for entry in questionnaire_entries if entry is not None]
                        )(
                            # Process lines with configurable trial context
                            (lambda trial_context, tp: [
                                (lambda parts, trial_info: (
                                    # Configurable trial context detection (fully generic)
                                    any(marker in parts[0].strip().lower() for marker in tp['trial_markers']) and parts[1].strip().isdigit() and trial_info.update({"trial_number": parts[1].strip()}) or
                                    any(marker in parts[0].strip().lower() for marker in tp['trigger_markers']) and parts[1].strip().isdigit() and trial_info.update({"trigger": parts[1].strip()}) or
                                    any(marker in parts[0].strip().lower() for marker in tp['procedure_markers']) and trial_info.update({"procedure": parts[1].strip()}) or
                                    # Configurable condition detection from pipeline patterns
                                    any(cond in parts[0].lower() for cond in tp['condition_markers']) and trial_info.update({
                                        "condition": (
                                            "positive" if any(pos in parts[1].upper() for pos in tp['positive_patterns']) else
                                            "negative" if any(neg in parts[1].upper() for neg in tp['negative_patterns']) else
                                            "neutral" if any(neu in parts[1].upper() for neu in tp['neutral_patterns']) else
                                            parts[1].strip()
                                        ),
                                        "stimulus_file": parts[1].strip()
                                    }) or
                                    # Return questionnaire data with trial context (if available)
                                    (
                                        any(quest in parts[0].lower() for quest in ['panas', 'bis', 'bas', 'sam', 'ea11', 'be7']) or
                                        any(pattern in parts[0].lower() for pattern in ['response', 'answer', 'rating', 'score', 'resp', 'choice', 'word']) or
                                        any(pattern in parts[0].lower() for pattern in ['rt', 'acc', 'onset', 'duration']) or
                                        any(pattern in parts[0].lower() for pattern in ['leftscale', 'rightscale', 'scale']) or
                                        # Include any key-value pair for maximum compatibility
                                        True
                                    ) and {
                                        "key": parts[0].strip(), 
                                        "value": parts[1].strip(),
                                        # Optional trial context (only added if detected)
                                        **({"trial_number": trial_info.get("trial_number")} if trial_info.get("trial_number") else {}),
                                        **({"condition": trial_info.get("condition")} if trial_info.get("condition") else {}),
                                        **({"stimulus_file": trial_info.get("stimulus_file")} if trial_info.get("stimulus_file") else {}),
                                        **({"trigger": trial_info.get("trigger")} if trial_info.get("trigger") else {}),
                                        **({"procedure": trial_info.get("procedure")} if trial_info.get("procedure") else {})
                                    }
                                ))(line.split(':', 1), trial_context) if ':' in line and len(line.split(':', 1)) == 2 else None
                                for line in text_lines
                                if line is not None and isinstance(line, str) and not line.strip().startswith('***') and not line.strip().endswith('***')
                            ])(
                                {}, 
                                trial_params
                            ))
                    )
                )
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
            # Parse trial detection parameters from pipeline (fully generic)
            if len(args) < 12:
                print(f"[PREPROC] ERROR: Missing parameters. Expected 11 arguments, got {len(args)-1}")
                usage()
            trial_params = {
                'trial_markers': args[4].split(','),
                'trigger_markers': args[5].split(','), 
                'procedure_markers': args[6].split(','),
                'condition_markers': args[7].split(','),
                'positive_patterns': args[8].split(','),
                'negative_patterns': args[9].split(','),
                'neutral_patterns': args[10].split(','),
                'aggregate_within_conditions': args[11].lower() == 'true'
            }
            run(input_parquet, encoding, output_dir, trial_params)
    except Exception as e:
        print(f"[PREPROC] Error: {e}")
        sys.exit(1)