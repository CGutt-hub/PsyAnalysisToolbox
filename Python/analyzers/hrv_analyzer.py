import polars as pl, numpy as np, sys, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python hrv_analyzer.py <input_parquet> <sfreq>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_hrv.parquet"
    run = lambda input_parquet, sfreq: (
        print(f"[Nextflow] HRV analysis started for input: {input_parquet}") or (
            # Lambda: read input data using Polars
            (lambda df:
                # Lambda: extract R-peak locations (supports two common column names)
                (lambda rpeaks:
                    # Lambda: validate R-peak data and compute RR intervals
                    (lambda rr_intervals:
                        # Only compute metrics if RR intervals are valid
                        (
                            (lambda metrics:
                                (pl.DataFrame(metrics).write_parquet(get_output_filename(input_parquet)),
                                 print(f"[Nextflow] HRV analysis finished for input: {input_parquet}"))
                            )([
                                {'metric': 'mean_rr', 'value': np.mean(rr_intervals)},      # Mean RR interval
                                {'metric': 'sdnn', 'value': np.std(rr_intervals)},          # Standard deviation of RR intervals
                                {'metric': 'rmssd', 'value': np.sqrt(np.mean(np.diff(rr_intervals) ** 2))}  # RMSSD
                            ])
                        ) if rr_intervals is not None else (
                            print(f"[Nextflow] HRV analysis errored for input: {input_parquet}. No R-peak column found or not enough peaks."),
                            pl.DataFrame([]).write_parquet(get_output_filename(input_parquet)),
                            sys.exit(1)
                        )
                    )(np.diff(rpeaks) / sfreq if rpeaks is not None and len(rpeaks) > 1 else None)
                )(df['R_Peak_Sample'].to_numpy() if 'R_Peak_Sample' in df.columns else df['rpeaks'].to_numpy() if 'rpeaks' in df.columns else None)
            )(pl.read_parquet(input_parquet).to_pandas())
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, sfreq = args[1], float(args[2])
            run(input_parquet, sfreq)
    except Exception as e:
        print(f"[Nextflow] HRV analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}")
        sys.exit(1)
