import polars as pl, numpy as np, sys, os
from scipy.signal import butter, filtfilt
if __name__ == "__main__":
    usage = lambda: print("[PROC] Usage: python filtering_processor.py <input_parquet> <filter_type> <frequencies> <sampling_freq> <data_columns>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_filtered.parquet"
    
    run = lambda input_parquet, filter_type, frequencies, sampling_freq, data_columns: (
        print(f"[PROC] Generic filtering started for: {input_parquet}") or
        (lambda df:
            (lambda fs:
                (lambda freq_list: 
                    (lambda column_list:
                        (lambda filtered_df:
                            (filtered_df.write_parquet(get_output_filename(input_parquet)),
                             print(f"[PROC] Generic filtering finished. Output: {get_output_filename(input_parquet)}"))
                        )(
                            # Apply filtering to each specified data column
                            df.with_columns([
                                pl.col(col).map_elements(
                                    lambda signal_series: (
                                        (lambda signal_array:
                                            (lambda butter_params:
                                                (lambda b, a:
                                                    filtfilt(b, a, signal_array) if isinstance(b, np.ndarray) and isinstance(a, np.ndarray) else signal_array
                                                )(butter_params[0], butter_params[1]) if butter_params is not None else signal_array
                                            )(
                                                # Generic butter filter configuration
                                                (lambda ftype, freqs, fs:
                                                    butter(2, freqs, btype=ftype, fs=fs) if (
                                                        ftype == 'low' and len(freqs) == 1 and 0 < freqs[0] < fs/2
                                                    ) or (
                                                        ftype == 'high' and len(freqs) == 1 and 0 < freqs[0] < fs/2  
                                                    ) or (
                                                        ftype == 'band' and len(freqs) == 2 and 0 < freqs[0] < freqs[1] < fs/2
                                                    ) or (
                                                        ftype == 'bandstop' and len(freqs) == 2 and 0 < freqs[0] < freqs[1] < fs/2
                                                    ) else None
                                                )(filter_type, freq_list, fs)
                                            )
                                        )(np.asarray(signal_series.to_list(), dtype=np.float64) if hasattr(signal_series, 'to_list') else np.asarray([signal_series], dtype=np.float64))
                                    ),
                                    return_dtype=pl.List(pl.Float64)
                                ).alias(f"{col}_filtered") for col in column_list
                            ])
                        )
                    )(data_columns.split(',') if isinstance(data_columns, str) else data_columns)
                )(
                    # Parse frequency parameters based on filter type
                    [float(f) for f in frequencies.split(',')] if isinstance(frequencies, str) else frequencies
                )
            )(float(sampling_freq))
        )(pl.read_parquet(input_parquet))
    )
    
    try:
        args = sys.argv
        if len(args) < 6:
            usage()
        else:
            input_parquet, filter_type, frequencies, sampling_freq, data_columns = args[1], args[2], args[3], args[4], args[5]
            run(input_parquet, filter_type, frequencies, sampling_freq, data_columns)
    except Exception as e:
        print(f"[PROC] Error: {e}")
        sys.exit(1)