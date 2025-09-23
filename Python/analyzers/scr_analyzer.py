import polars as pl, sys
if __name__ == "__main__":
    usage = lambda: print("Usage: python scr_analyzer.py <input_parquet> <participant_id> [output_parquet]") or sys.exit(1)
    run = lambda input_parquet, participant_id, output_parquet: (
        print(f"[Nextflow] SCR analysis started for participant: {participant_id}") or (
            # Lambda: read EDA data from Parquet
            (lambda eda_df:
                # Lambda: compute SCR metrics (placeholder, implement actual SCR computation)
                (lambda scr_results:
                    # Lambda: convert results to Polars DataFrame and write to Parquet
                    (pl.DataFrame(scr_results).write_parquet(output_parquet),
                     print(f"[Nextflow] SCR analysis finished for participant: {participant_id}"))
                )([
                    # Placeholder SCR metrics for demonstration
                    {'event': None, 'amplitude': None, 'latency': None}
                ]) if eda_df is not None and len(eda_df) > 0 else (
                    print(f"[Nextflow] SCR analysis errored for participant: {participant_id}. No EDA data found."),
                    pl.DataFrame([]).write_parquet(output_parquet),
                    sys.exit(1)
                )
            )(pl.read_parquet(input_parquet).to_pandas())
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, participant_id = args[1], args[2]
            output_parquet = args[3] if len(args) > 3 else f"{participant_id}_scr.parquet"
            run(input_parquet, participant_id, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] SCR analysis errored for participant: {pid}. Error: {e}")
        sys.exit(1)