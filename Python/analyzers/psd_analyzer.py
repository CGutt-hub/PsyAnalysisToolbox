import polars as pl, numpy as np, mne, sys, ast, os

# Load epoched data from parquet and convert to MNE epochs
load_epochs_from_parquet = lambda input_parquet: (
    (lambda df: (
        (lambda ch_names: (
            (lambda sfreq: (
                (lambda epoch_groups: (
                    mne.EpochsArray(
                        np.stack([group.select(ch_names).to_numpy().T for group in epoch_groups]),
                        mne.create_info(ch_names, sfreq, ch_types='eeg'),
                        events=np.column_stack([
                            np.arange(len(epoch_groups)),
                            np.zeros(len(epoch_groups), dtype=int),
                            [group.select('event').to_series()[0] for group in epoch_groups]
                        ]).astype(int),
                        tmin=-1.0,
                        verbose=False
                    )
                ))([df.filter(pl.col('epoch_id') == epoch_id) for epoch_id in df.select('epoch_id').unique()])
            ))(df['sfreq'][0] if 'sfreq' in df.columns else 1000.0)
        ))([col for col in df.columns if col not in ['time', 'sfreq', 'data_type', 'event', 'epoch_id']])
    ))(pl.read_parquet(input_parquet))
)

if __name__ == "__main__":
    usage = lambda: print("Usage: python psd_analyzer.py <input_parquet> <bands_config> [channels_config]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_psd.parquet"
    
    run = lambda input_parquet, bands_config, channels_config: (
        print(f"[PSD] PSD analysis started for input: {input_parquet}") or (
            # Load epochs from parquet
            (lambda epochs: (
                # Parse bands configuration
                (lambda bands: (
                    # Select channels of interest
                    (lambda channels: (
                        # Compute PSD using MNE's scientific method
                        (lambda psd_result: (
                            # Extract PSD data and frequencies
                            (lambda psd_tuple: (
                                (lambda psd_data, freqs: (
                                # Compute band power for each channel
                                (lambda band_powers: (
                                    pl.DataFrame([
                                        {
                                            'channel': ch_name,
                                            'band': band_name,
                                            'power': float(power),
                                            'frequency_range': f"{band_range[0]}-{band_range[1]}Hz",
                                            'plot_type': 'bar',
                                            'x_scale': 'ordinal', 
                                            'y_scale': 'nominal',
                                            'x_data': f"{ch_name}_{band_name}",
                                            'y_data': float(power),
                                            'y_label': 'Power (μV²/Hz)',
                                            'plot_weight': 1,
                                            'data_type': 'analysis_result'
                                        }
                                        for ch_idx, ch_name in enumerate(epochs.ch_names)
                                        for band_name, (power, band_range) in band_powers[ch_idx].items()
                                    ]).write_parquet(get_output_filename(input_parquet)),
                                    print(f"[PSD] PSD analysis finished for input: {input_parquet}")
                                ))([
                                    {
                                        band_name: (
                                            float(np.mean(psd_data[:, ch_idx, np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]])),
                                            band_range
                                        )
                                        for band_name, band_range in bands.items()
                                    }
                                    for ch_idx in range(len(epochs.ch_names))
                                ])
                            ))(psd_tuple[0], psd_tuple[1])
                        ))(psd_result.get_data())
                    ))(epochs.compute_psd(
                            method='welch',
                            fmin=min(band_range[0] for band_range in bands.values()),
                            fmax=max(band_range[1] for band_range in bands.values()),
                            picks=channels if channels else 'all',
                            verbose=False
                        ))
                    ))(ast.literal_eval(channels_config) if channels_config and channels_config != 'None' else None)
                ))(ast.literal_eval(bands_config) if isinstance(bands_config, str) else bands_config)
            ))(load_epochs_from_parquet(input_parquet))
        )
    )

    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet = args[1]
            bands_config = args[2]
            channels_config = args[3] if len(args) > 3 else None
            run(input_parquet, bands_config, channels_config)
    except Exception as e:
        print(f"[PSD] PSD analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}")
        sys.exit(1)
