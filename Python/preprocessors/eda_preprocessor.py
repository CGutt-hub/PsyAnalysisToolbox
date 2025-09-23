import polars as pl, numpy as np, sys

# Ultra-compressed, lambda-driven EDA preprocessor for generic EDA signal cleaning
if __name__ == "__main__":
    usage = lambda: print("Usage: python eda_preprocessor.py <input_parquet> <participant_id> [output_parquet]") or sys.exit(1)
    run = lambda input_parquet, participant_id, output_parquet: (
        print(f"[Nextflow] EDA preprocessing started for participant: {participant_id}") or (
            # Lambda: read EDA signal from Parquet
            (lambda df:
                # Lambda: extract EDA signal
                (lambda eda_signal:
                    # Lambda: check for valid EDA signal
                    (
                        (pl.DataFrame([{'eda': v, 'participant_id': participant_id} for v in eda_signal]).write_parquet(output_parquet),
                         print(f"[Nextflow] EDA preprocessing finished for participant: {participant_id}"))
                    ) if (isinstance(eda_signal, np.ndarray) and not np.isnan(eda_signal).any() and eda_signal.size > 0)
                    else (
                        print(f"[Nextflow] EDA preprocessing errored for participant: {participant_id}. Invalid or missing EDA signal."),
                        pl.DataFrame([]).write_parquet(output_parquet),
                        sys.exit(1)
                    )
                )(df['eda'].to_numpy() if 'eda' in df.columns else None)
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, participant_id = args[1], args[2]
            output_parquet = args[3] if len(args) > 3 else f"{participant_id}_eda.parquet"
            run(input_parquet, participant_id, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] EDA preprocessing errored for participant: {pid}. Error: {e}")
        sys.exit(1)