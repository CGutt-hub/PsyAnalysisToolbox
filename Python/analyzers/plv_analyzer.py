import polars as pl, numpy as np, sys, os
from scipy.signal import hilbert
if __name__ == "__main__":
    usage = lambda: print("Usage: python plv_analyzer.py <input_parquet>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_plv.parquet"
    run = lambda input_parquet: (
        print(f"[Nextflow] PLV analysis started for input: {input_parquet}") or (
            # Adaptive PLV: Pairwise for 2 channels, Multi-channel for >2 channels
            (lambda df:
                (lambda results:
                    pl.DataFrame(results).write_parquet(get_output_filename(input_parquet)) or
                    print(f"[Nextflow] PLV analysis finished for input: {input_parquet}")
                )(
                    # Lambda: calculate multi-channel PLV for all channels together
                    (lambda channels:
                        (lambda signals:
                            (lambda analytic_signals:
                                (lambda phases:
                                    # If only 2 channels: pairwise PLV
                                    [{
                                        'channel1': channels[0],
                                        'channel2': channels[1],
                                        'plv_value': float(np.abs(np.mean(np.exp(1j * (phases[0] - phases[1]))))),
                                        'plot_type': 'bar',
                                        'x_scale': 'ordinal',
                                        'y_scale': 'nominal',
                                        'x_data': f"{channels[0]}-{channels[1]}",
                                        'y_data': float(np.abs(np.mean(np.exp(1j * (phases[0] - phases[1]))))),
                                        'y_label': 'PLV (0-1)',
                                        'plot_weight': 1,
                                        'data_type': 'analysis_result'
                                    }] if len(channels) == 2 else
                                    # If >2 channels: multi-channel PLV
                                    [{
                                        'channels': '_'.join(channels),
                                        'num_channels': len(channels),
                                        'plv_value': float(np.abs(np.mean(np.exp(1j * (phases - np.mean(phases, axis=0)).mean(axis=0))))),
                                        'plot_type': 'bar',
                                        'x_scale': 'ordinal',
                                        'y_scale': 'nominal',
                                        'x_data': f"{len(channels)}_channel_PLV",
                                        'y_data': float(np.abs(np.mean(np.exp(1j * (phases - np.mean(phases, axis=0)).mean(axis=0))))),
                                        'y_label': 'Multi-Channel PLV (0-1)',
                                        'plot_weight': 1,
                                        'data_type': 'analysis_result'
                                    }]
                                )(np.array([np.angle(np.asarray(sig, dtype=complex)) for sig in analytic_signals]))
                            )([hilbert(sig) for sig in signals])
                        )([df[ch].to_numpy() for ch in channels])
                    )([ch for ch in df.columns if ch not in ['time', 'sfreq', 'data_type']])
                )
            )(pl.read_parquet(input_parquet))
        )
    )
    
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            run(input_parquet)
    except Exception as e:
        print(f"[Nextflow] PLV analysis errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)