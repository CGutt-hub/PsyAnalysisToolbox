import polars as pl, numpy as np, neurokit2 as nk, sys
if __name__ == "__main__":
    usage = lambda: print("Usage: python ecg_preprocessor.py <input_parquet> <participant_id> [sampling_rate] [output_parquet]") or sys.exit(1)
    run = lambda input_parquet, participant_id, sampling_rate, output_parquet: (
        print(f"[Nextflow] ECG preprocessing started for participant: {participant_id}") or (
            # Lambda: read ECG signal from Parquet
            (lambda df:
                # Lambda: extract ECG signal
                (lambda ecg_signal:
                    # Lambda: check for valid ECG signal
                    (lambda valid:
                        # Lambda: run NeuroKit2 R-peak detection
                        (lambda rpeaks:
                            # Lambda: write results to Parquet
                            (pl.DataFrame([{'R_Peak_Sample': r, 'participant_id': participant_id} for r in rpeaks]).write_parquet(output_parquet),
                             print(f"[Nextflow] ECG preprocessing finished for participant: {participant_id}"))
                        )(nk.ecg_findpeaks(ecg_signal, sampling_rate=sampling_rate)['ECG_R_Peaks'] if valid else (
                            print(f"[Nextflow] ECG preprocessing errored for participant: {participant_id}. Invalid ECG signal."),
                            pl.DataFrame([]).write_parquet(output_parquet),
                            sys.exit(1)
                        ))
                    )(isinstance(ecg_signal, np.ndarray) and not np.isnan(ecg_signal).any() and ecg_signal.size > 0)
                )(df['ecg'].to_numpy() if 'ecg' in df.columns else None)
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, participant_id = args[1], args[2]
            sampling_rate = int(args[3]) if len(args) > 3 else 1000
            output_parquet = args[4] if len(args) > 4 else f"{participant_id}_ecg.parquet"
            run(input_parquet, participant_id, sampling_rate, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] ECG preprocessing errored for participant: {pid}. Error: {e}")
        sys.exit(1)