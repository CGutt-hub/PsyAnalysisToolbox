import polars as pl, numpy as np, sys, os
from scipy.signal import hilbert
if __name__ == "__main__":
    usage = lambda: print("Usage: python plv_analyzer.py <input_parquet>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_plv.parquet"
    run = lambda input_parquet: (
        print(f"[PLV] PLV analysis started for input: {input_parquet}") or (
            # Adaptive PLV: Pairwise for 2 channels, Multi-channel for >2 channels
            (lambda df:
                (lambda results:
                    pl.DataFrame(results).write_parquet(get_output_filename(input_parquet)) or
                    print(f"[PLV] PLV analysis finished for input: {input_parquet}")
                )(
                    # Lambda: calculate multi-channel PLV for all channels together
                    (lambda channels:
                        (lambda signals:
                            (lambda analytic_signals:
                                (lambda phases:
                                    # If only 2 channels: pairwise PLV + temporal dynamics
                                    (lambda overall_plv, temporal_plv: [{
                                        'channel1': channels[0],
                                        'channel2': channels[1],
                                        'plv_value': float(overall_plv),
                                        'analysis_type': 'overall',
                                        'plot_type': 'bar',
                                        'x_scale': 'ordinal',
                                        'y_scale': 'nominal',
                                        'x_data': f"{channels[0]}-{channels[1]}",
                                        'y_data': float(overall_plv),
                                        'y_label': 'PLV (0-1)',
                                        'plot_weight': 1,
                                        'data_type': 'analysis_result'
                                    }] + temporal_plv)(
                                        # Overall PLV (original)
                                        np.abs(np.mean(np.exp(1j * (phases[0] - phases[1])))),
                                        # Temporal dynamics PLV (sliding window - as per proposal)
                                        (lambda window_size: [
                                            {
                                                'channel1': channels[0],
                                                'channel2': channels[1],
                                                'plv_value': float(np.abs(np.mean(np.exp(1j * (phases[0][i:i+window_size] - phases[1][i:i+window_size]))))),
                                                'analysis_type': 'temporal',
                                                'time_window': i,
                                                'window_center_time': i + window_size//2,
                                                'plot_type': 'line',
                                                'x_scale': 'continuous',
                                                'y_scale': 'continuous',
                                                'x_data': i + window_size//2,
                                                'y_data': float(np.abs(np.mean(np.exp(1j * (phases[0][i:i+window_size] - phases[1][i:i+window_size]))))),
                                                'y_label': 'PLV (0-1)',
                                                'plot_weight': 0.5,
                                                'data_type': 'temporal_dynamics'
                                            }
                                            for i in range(0, len(phases[0]) - window_size + 1, window_size//4)  # 75% overlap
                                        ] if len(phases[0]) > window_size else [])(
                                            min(len(phases[0])//4, 250)  # Window size: 1/4 of signal or max 250 samples
                                        )
                                    ) if len(channels) == 2 else
                                    # If >2 channels: multi-channel PLV + temporal dynamics
                                    (lambda overall_plv, temporal_plv: [{
                                        'channels': '_'.join(channels),
                                        'num_channels': len(channels),
                                        'plv_value': float(overall_plv),
                                        'analysis_type': 'overall',
                                        'plot_type': 'bar',
                                        'x_scale': 'ordinal',
                                        'y_scale': 'nominal',
                                        'x_data': f"{len(channels)}_channel_PLV",
                                        'y_data': float(overall_plv),
                                        'y_label': 'Multi-Channel PLV (0-1)',
                                        'plot_weight': 1,
                                        'data_type': 'analysis_result'
                                    }] + temporal_plv)(
                                        # Overall multi-channel PLV (original)
                                        np.abs(np.mean(np.exp(1j * (phases - np.mean(phases, axis=0)).mean(axis=0)))),
                                        # Temporal dynamics for multi-channel (sliding window)
                                        (lambda window_size: [
                                            {
                                                'channels': '_'.join(channels),
                                                'num_channels': len(channels),
                                                'plv_value': float(np.abs(np.mean(np.exp(1j * (phases[:, i:i+window_size] - np.mean(phases[:, i:i+window_size], axis=0)).mean(axis=0))))),
                                                'analysis_type': 'temporal',
                                                'time_window': i,
                                                'window_center_time': i + window_size//2,
                                                'plot_type': 'line',
                                                'x_scale': 'continuous',
                                                'y_scale': 'continuous',
                                                'x_data': i + window_size//2,
                                                'y_data': float(np.abs(np.mean(np.exp(1j * (phases[:, i:i+window_size] - np.mean(phases[:, i:i+window_size], axis=0)).mean(axis=0))))),
                                                'y_label': 'Multi-Channel PLV (0-1)',
                                                'plot_weight': 0.5,
                                                'data_type': 'temporal_dynamics'
                                            }
                                            for i in range(0, phases.shape[1] - window_size + 1, window_size//4)  # 75% overlap
                                        ] if phases.shape[1] > window_size else [])(
                                            min(phases.shape[1]//4, 250)  # Window size: 1/4 of signal or max 250 samples
                                        )
                                    )
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
        print(f"[PLV] PLV analysis errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)