import polars as pl, sys
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python epoch_processor.py <input_parquet> <event_parquet> <pre_ms> <post_ms> <output_parquet>") or sys.exit(1)
    # Lambda: read signal and event Parquet files
    read_parquet = lambda f: pl.read_parquet(f)
    # Lambda: main epoching logic (generic for any signal type)
    run = lambda input_parquet, event_parquet, pre_ms, post_ms, output_parquet: (
        print(f"[Nextflow] Epoching started for: {input_parquet}") or (
            # Lambda: read signal and event data
            (lambda signal, events: (
                print(f"[Nextflow] Loaded signal shape: {signal.shape}, events shape: {events.shape}"),
                # Lambda: create epochs around each event
                (lambda epochs: (
                    print(f"[Nextflow] Created {epochs.height} epochs."),
                    # Lambda: write epochs to Parquet
                    epochs.write_parquet(output_parquet),
                    print(f"[Nextflow] Epoching finished. Output: {output_parquet}")
                ))(pl.concat([
                    signal.filter((pl.col('time') >= ev['time'] - pre_ms) & (pl.col('time') <= ev['time'] + post_ms)).with_columns([pl.lit(ev['event']).alias('event')])
                    for ev in events.iter_rows(named=True)
                ]) if events.height > 0 else pl.DataFrame([]))
            ))(read_parquet(input_parquet), read_parquet(event_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 6:
            usage()
        else:
            input_parquet, event_parquet = args[1], args[2]
            pre_ms, post_ms = int(args[3]), int(args[4])
            output_parquet = args[5]
            run(input_parquet, event_parquet, pre_ms, post_ms, output_parquet)
    except Exception as e:
        print(f"[Nextflow] Epoching errored. Error: {e}")
        sys.exit(1)