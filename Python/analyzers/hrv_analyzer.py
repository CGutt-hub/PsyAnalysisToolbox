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
                            (lambda processed_data, metrics:
                                (pl.concat([
                                    # RR intervals for downstream analysis
                                    pl.DataFrame(processed_data).with_columns([
                                        pl.lit("processed_rr").alias("data_type"),
                                        pl.lit(None).alias("plot_type")  # Processed data, not for plotting
                                    ]),
                                    # HRV metrics for plotting
                                    pl.DataFrame(metrics).with_columns([
                                        pl.lit("analysis_result").alias("data_type")
                                    ])
                                ]).write_parquet(get_output_filename(input_parquet)),
                                 print(f"[Nextflow] HRV analysis finished for input: {input_parquet}"))
                            )(
                                # RR intervals for downstream analysis
                                [{'rr_interval': interval, 'sample_idx': idx} for idx, interval in enumerate(rr_intervals)],
                                # HRV analysis metrics
                                [
                                    {
                                        'metric': 'mean_rr', 'value': np.mean(rr_intervals),
                                        'plot_type': 'bar', 'x_scale': 'ordinal', 'y_scale': 'nominal',
                                        'x_data': 'mean_rr', 'y_data': np.mean(rr_intervals), 
                                        'y_label': 'Time (ms)', 'plot_weight': 1
                                    },
                                    {
                                        'metric': 'sdnn', 'value': np.std(rr_intervals),
                                        'plot_type': 'bar', 'x_scale': 'ordinal', 'y_scale': 'nominal',
                                        'x_data': 'sdnn', 'y_data': np.std(rr_intervals), 
                                        'y_label': 'Time (ms)', 'plot_weight': 1
                                    },
                                    {
                                        'metric': 'rmssd', 'value': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
                                        'plot_type': 'bar', 'x_scale': 'ordinal', 'y_scale': 'nominal',
                                    'x_data': 'rmssd', 'y_data': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)), 
                                    'y_label': 'Time (ms)', 'plot_weight': 1
                                }
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
