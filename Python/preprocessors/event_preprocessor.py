import polars as pl, sys, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: (print("[PREPROC] Usage: python event_preprocessor.py <events_parquet> <sfreq> <trigger_condition_map>"), sys.exit(1))
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_event.parquet"
    run = lambda events_parquet, sfreq, trigger_condition_map: (
        print(f"[PREPROC] Event handling started for: {events_parquet}") or
        (lambda df:
            print(f"[PREPROC] Loaded events DataFrame shape: {df.shape}") or
            (lambda trig_map:
                print(f"[PREPROC] Using trigger-condition map: {trig_map}") or
                (lambda events:
                    print(f"[PREPROC] Created standardized events shape: {events.shape}") or
                    (events.write_parquet(get_output_filename(events_parquet)), print(f"[PREPROC] Event handling finished. Output: {get_output_filename(events_parquet)}"))
                )(
                    df.with_columns([
                        (pl.col('onset') * float(sfreq)).cast(int).alias('onset_sample'),
                        pl.col('trigger'),
                        pl.col('trigger').map_elements(lambda x: trig_map.get(str(x), 'unknown')).alias('condition'),
                        pl.col('trigger').map_elements(lambda x: int(x)).alias('event_id'),
                        pl.col('onset').alias('time'),
                        pl.lit(float(sfreq)).alias('sfreq'),
                        pl.lit('preprocessed_events').alias('data_type')
                    ]) if 'onset' in df.columns and 'trigger' in df.columns else pl.DataFrame([])
                )
            )(dict(item.split(':') for item in trigger_condition_map.split(',') if ':' in item))
        )(pl.read_parquet(events_parquet))
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            events_parquet, sfreq, trigger_condition_map = args[1], args[2], args[3]
            run(events_parquet, sfreq, trigger_condition_map)
    except Exception as e:
        print(f"[PREPROC] Error: {e}")
        sys.exit(1)