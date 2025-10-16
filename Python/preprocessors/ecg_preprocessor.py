import polars as pl, numpy as np, neurokit2 as nk, sys, os
if __name__ == "__main__":
    usage = lambda: print("[PREPROC] Usage: python ecg_preprocessor.py <input_parquet> [sampling_rate]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_ecg.parquet"
    run = lambda input_parquet, sampling_rate: (
        print(f"[PREPROC] ECG preprocessing started for file: {input_parquet}") or (
            (lambda df:
                (lambda ecg_signal:
                    (lambda valid:
                        (lambda rpeaks:
                            (pl.DataFrame([
                                {'R_Peak_Sample': r, 'time': r/sampling_rate, 'sfreq': sampling_rate, 'data_type': 'preprocessed_ecg'} 
                                for r in rpeaks
                            ]).write_parquet(get_output_filename(input_parquet)),
                             print(f"[PREPROC] ECG preprocessing finished for file: {input_parquet}"))
                        )(nk.ecg_findpeaks(ecg_signal, sampling_rate=sampling_rate)['ECG_R_Peaks'] if valid else (
                            print(f"[PREPROC] ECG preprocessing errored for file: {input_parquet}. Invalid ECG signal."),
                            pl.DataFrame([]).write_parquet(get_output_filename(input_parquet)),
                            sys.exit(1)
                        ))
                    )(isinstance(ecg_signal, np.ndarray) and not np.isnan(ecg_signal).any() and ecg_signal.size > 0)
                )(df['ecg'].to_numpy() if 'ecg' in df.columns else None)
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            sampling_rate = float(args[2]) if len(args) > 2 else 1000.0
            run(input_parquet, sampling_rate)
    except Exception as e:
        print(f"[PREPROC] Error: {e}")
        sys.exit(1)