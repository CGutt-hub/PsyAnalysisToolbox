import polars as pl, sys, re, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python questionnaire_analyzer.py <input_parquet> <question_pattern> <response_pattern> <scale_pattern>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_quest.parquet"
    run = lambda input_parquet, question_pattern, response_pattern, scale_pattern: (
        print(f"[Nextflow] Questionnaire analysis started for: {input_parquet}") or (
            (lambda df:
                print(f"[Nextflow] Loaded questionnaire DataFrame shape: {df.shape}") or (
                    'key' not in df.columns and (
                        print("[Nextflow] Error: Expected key-value format with 'key' column not found") or sys.exit(1)
                    ) or
                    (lambda matching_keys:
                        print(f"[Nextflow] Found matching keys: {len(matching_keys)}") or
                        print(f"[Nextflow] Sample matching keys: {matching_keys[:5] if matching_keys else 'None'}") or (
                            (lambda filtered_df:
                                print(f"[Nextflow] Filtered data shape: {filtered_df.shape}") or
                                (lambda structured_df:
                                    print(f"[Nextflow] Structured data shape: {structured_df.shape}") or
                                    print(f"[Nextflow] Structured columns: {structured_df.columns}") or (
                                        structured_df.write_parquet(get_output_filename(input_parquet)),
                                        print(f"[Nextflow] Questionnaire analysis finished. Output: {get_output_filename(input_parquet)}")
                                    )
                                )(
                                    # Generic questionnaire analysis with configurable patterns
                                    (lambda question_keys, scale_keys, choice_keys:
                                        print(f"[Nextflow] Found {len(question_keys)} question items, {len(scale_keys)} scale items, {len(choice_keys)} choice responses") or
                                        (lambda structured_data:
                                            pl.DataFrame(structured_data) if structured_data else pl.DataFrame({
                                                "questionnaire_type": [], "item_id": [], "question_text": [],
                                                "response_value": [], "scale_text": [], "scale_numeric": [],
                                                "x_axis": [], "y_axis": [], "plot_weight": []
                                            })
                                        )([
                                            {
                                                # Extract questionnaire type from key pattern
                                                "questionnaire_type": (lambda match: match.group(0) if match else "unknown")(re.search(question_pattern, choice["key"], re.IGNORECASE) or re.search(response_pattern, choice["key"], re.IGNORECASE) or re.search(scale_pattern, choice["key"], re.IGNORECASE)),
                                                # Item identifier for grouping
                                                "item_id": choice["key"],
                                                # Find corresponding question text
                                                "question_text": next((q["value"] for q in question_keys 
                                                                     if any(part.lower() in choice["key"].lower() for part in q["key"].split())), 
                                                                     choice["key"]),
                                                # Raw response value
                                                "response_value": choice["value"],
                                                # Scale text from matching scale key
                                                "scale_text": next((
                                                    s["value"] for s in scale_keys 
                                                    if s["key"].endswith(choice["value"]) or choice["value"] in s["key"]
                                                ), ""),
                                                # Convert to numeric scale (configurable conversion)
                                                "scale_numeric": int(choice["value"]) if choice["value"].isdigit() else 0,
                                                # X-axis shows the questionnaire text from available question keys
                                                "x_axis": next((q["value"] for q in question_keys 
                                                              if any(part in choice["key"].lower() for part in q["key"].lower().split())), 
                                                             choice["key"].split(".")[0] if "." in choice["key"] else choice["key"]),
                                                # Y-axis label from scale text
                                                "y_axis": next((
                                                    s["value"] for s in scale_keys 
                                                    if s["key"].endswith(choice["value"]) or choice["value"] in s["key"]
                                                ), ""),
                                                # Plot metadata for generic plotter
                                                "plot_type": "bar",  # questionnaire responses are categorical -> bar chart
                                                "x_scale": "nominal",  # categories (questionnaire items)
                                                "y_scale": "ordinal",  # ordered response scale (0-4)
                                                "x_data": next((q["value"] for q in question_keys 
                                                              if any(part in choice["key"].lower() for part in q["key"].lower().split())), 
                                                             choice["key"].split(".")[0] if "." in choice["key"] else choice["key"]),
                                                "y_data": int(choice["value"]) if choice["value"].isdigit() else 0,
                                                # Weight for aggregation (typically 1 for counting)
                                                "plot_weight": 1
                                            }
                                            for choice in choice_keys
                                        ])
                                    )(
                                        # Extract question items based on question_pattern
                                        [{"key": row["key"], "value": row["value"]} 
                                         for row in filtered_df.to_dicts() 
                                         if re.search(question_pattern, row["key"], re.IGNORECASE) and len(row["value"]) > 5],
                                        # Extract scale items based on response_pattern and scale_pattern
                                        [{"key": row["key"], "value": row["value"]} 
                                         for row in filtered_df.to_dicts() 
                                         if re.search(response_pattern, row["key"], re.IGNORECASE) or re.search(scale_pattern, row["key"], re.IGNORECASE)],
                                        # Extract response choices (items with .Choice, .Value, numeric responses)
                                        [{"key": row["key"], "value": row["value"]} 
                                         for row in filtered_df.to_dicts() 
                                         if (re.search(question_pattern, row["key"], re.IGNORECASE) and 
                                             (".Choice" in row["key"] or ".Value" in row["key"] or row["value"].isdigit()))]
                                    )
                                ) if filtered_df.shape[0] > 0 else pl.DataFrame({
                                    "key": [], "value": [], "questionnaire_type": [], "item_number": [], 
                                    "is_numeric": [], "score": [], "x_label": [], "y_label": [], "scale_name": []
                                })
                            )(df.filter(pl.col("key").is_in(matching_keys)) if matching_keys else pl.DataFrame({"key": [], "value": []}))
                        )
                    )([
                        key for key in df["key"].unique().to_list()
                        if re.search(question_pattern, key, re.IGNORECASE) or re.search(response_pattern, key, re.IGNORECASE) or re.search(scale_pattern, key, re.IGNORECASE)
                    ])
                )
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        elif len(args) == 4:
            # Dispatcher format: quest_analyzer.py <input_parquet> "<question_pattern> <response_pattern> <scale_pattern>"
            input_parquet = args[1]
            patterns = args[2].split(' ', 2)
            if len(patterns) < 3:
                usage()
            else:
                question_pattern = patterns[0]
                response_pattern = patterns[1]
                scale_pattern = patterns[2]
                run(input_parquet, question_pattern, response_pattern, scale_pattern)
        else:
            # Direct format: quest_analyzer.py <input_parquet> <question_pattern> <response_pattern> <scale_pattern>
            input_parquet = args[1]
            question_pattern = args[2]
            response_pattern = args[3]
            scale_pattern = args[4]
            run(input_parquet, question_pattern, response_pattern, scale_pattern)
    except Exception as e:
        print(f"[Nextflow] Questionnaire analysis errored. Error: {e}")
        sys.exit(1)