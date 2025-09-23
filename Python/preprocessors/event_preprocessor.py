import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python event_handling_processor.py <events_parquet> <sfreq> <output_parquet>") or sys.exit(1)
    # Lambda: read events Parquet file
    read_parquet = lambda f: pl.read_parquet(f)
    # Lambda: map conditions to event IDs
    map_conditions = lambda conds: {str(cond): i+1 for i, cond in enumerate(conds)}
    # Lambda: create standardized events DataFrame
    create_events_df = lambda df, sfreq: df.with_columns([
        (pl.col('onset') * sfreq).cast(int).alias('onset_sample'),
        pl.col('condition'),
        pl.col('condition').map_elements(lambda x: map_conditions(df['condition'].unique())[x]).alias('event_id')
    ]) if 'onset' in df.columns and 'condition' in df.columns else pl.DataFrame([])
    # Lambda: write events DataFrame to Parquet
    write_parquet = lambda df, output_parquet: df.write_parquet(output_parquet)
    # Lambda: main event handling logic
    run = lambda events_parquet, sfreq, output_parquet: (
        print(f"[Nextflow] Event handling started for: {events_parquet}") or (
            # Lambda: read events
            (lambda df: (
                print(f"[Nextflow] Loaded events DataFrame shape: {df.shape}"),
                # Lambda: create standardized events DataFrame
                (lambda events: (
                    print(f"[Nextflow] Created standardized events shape: {events.shape}"),
                    # Lambda: write output Parquet
                    write_parquet(events, output_parquet),
                    print(f"[Nextflow] Event handling finished. Output: {output_parquet}")
                ))(create_events_df(df, float(sfreq)))
            ))(read_parquet(events_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            events_parquet, sfreq, output_parquet = args[1], args[2], args[3]
            run(events_parquet, sfreq, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Event handling errored. Error: {e}")
        sys.exit(1)