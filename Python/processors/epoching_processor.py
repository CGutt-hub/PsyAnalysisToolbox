import polars as pl, sys, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python epoching_processor.py <input_parquet> <event_parquet>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_epoch.parquet"
    run = lambda input_parquet, event_parquet: (
        print(f"[Nextflow] Epoching started for: {input_parquet}") or
        (lambda signal:
            (lambda events:
                (lambda epochs:
                    print(f"[Nextflow] Loaded signal shape: {signal.shape}, events shape: {events.shape}") or
                    print(f"[Nextflow] Created {epochs.height} epochs.") or
                    epochs.write_parquet(get_output_filename(input_parquet)) or
                    print(f"[Nextflow] Epoching finished. Output: {get_output_filename(input_parquet)}")
                )(
                    pl.concat([
                        signal.filter((pl.col('time') >= ev['time'] - 1.0) & (pl.col('time') <= ev['time'] + 1.0)).with_columns([
                            pl.lit(ev.get('event_id', ev.get('trigger', ev.get('condition', 'unknown')))).alias('event'),
                            pl.lit(f"epoch_{i}").alias('epoch_id')
                        ])
                        for i, ev in enumerate(events.iter_rows(named=True))
                    ]) if events.height > 0 else pl.DataFrame([])
                )
            )(pl.read_parquet(event_parquet))
        )(pl.read_parquet(input_parquet))
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, event_parquet = args[1], args[2]
            run(input_parquet, event_parquet)
    except Exception as e:
        print(f"[Nextflow] Epoching errored. Error: {e}")
        sys.exit(1)