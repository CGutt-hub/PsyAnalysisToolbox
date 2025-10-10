import polars as pl, numpy as np, sys, os, scipy.signal
if __name__ == "__main__":
    usage = lambda: print("Usage: python eda_preprocessor.py <input_parquet> <l_freq> <h_freq>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_eda.parquet"
    run = lambda input_parquet, l_freq, h_freq: (
        print(f"[Nextflow] EDA preprocessing started for file: {input_parquet}") or (
            (lambda df:
                (lambda eda_signal:
                    (
                        (lambda filtered:
                            (
                                print(f"[Nextflow] EDA preprocessing finished for file: {input_parquet}"),
                                pl.DataFrame([
                                    {'eda': v, 'time': i/1000.0, 'sfreq': 1000.0, 'data_type': 'preprocessed_eda'} 
                                    for i, v in enumerate(filtered)
                                ]).write_parquet(get_output_filename(input_parquet))
                            ) if (isinstance(filtered, np.ndarray) and not np.isnan(filtered).any() and filtered.size > 0)
                            else (
                                print(f"[Nextflow] EDA preprocessing errored for file: {input_parquet}. Invalid or missing EDA signal."),
                                pl.DataFrame([]).write_parquet(get_output_filename(input_parquet)),
                                sys.exit(1)
                            )
                        )(
                            (lambda s, lf, hf:
                                (lambda lf_val, hf_val:
                                    (lambda butter_result:
                                        (lambda b, a:
                                            scipy.signal.filtfilt(b, a, s)
                                            if isinstance(b, np.ndarray) and isinstance(a, np.ndarray)
                                            else s
                                        )(butter_result[0], butter_result[1])
                                        if isinstance(butter_result, tuple) and len(butter_result) == 2
                                        else s
                                    )(scipy.signal.butter(2, [lf_val, hf_val], btype='band', fs=1000))
                                    if (
                                        lf_val is not None and hf_val is not None
                                        and 0 < lf_val < hf_val < 500
                                        and isinstance(s, np.ndarray) and s.size > 0
                                    ) else s
                                )(
                                    float(lf) if lf is not None and str(lf).replace('.', '', 1).isdigit() else None,
                                    float(hf) if hf is not None and str(hf).replace('.', '', 1).isdigit() else None
                                ) if lf is not None and hf is not None else s
                            )(eda_signal, l_freq, h_freq)
                        )
                    ) if (eda_signal is not None and isinstance(eda_signal, np.ndarray)) else (
                        print(f"[Nextflow] EDA preprocessing errored for file: {input_parquet}. Invalid or missing EDA signal."),
                        pl.DataFrame([]).write_parquet(get_output_filename(input_parquet)),
                        sys.exit(1)
                    )
                )(df['eda'].to_numpy() if 'eda' in df.columns else None)
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            l_freq = float(args[2]) if len(args) > 2 else 0.05
            h_freq = float(args[3]) if len(args) > 3 else 5.0
            run(input_parquet, l_freq, h_freq)
    except Exception as e:
        print(f"[Nextflow] EDA preprocessing errored for file: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)