import polars as pl, numpy as np, mne, sys
if __name__ == "__main__":
    import os
    usage = lambda: print("Usage: python psd_analyzer.py <input_fif> <bands_parquet> [channels_parquet]") or sys.exit(1)
    get_output_filename = lambda input_fif: f"{os.path.splitext(os.path.basename(input_fif))[0]}_psd.parquet"
    run = lambda input_fif, bands_parquet, channels_parquet: (
        print(f"[Nextflow] PSD analysis started for input: {input_fif}") or (
            # Lambda: read epochs from FIF file
            (lambda epochs:
                # Lambda: read bands config from Parquet
                (lambda bands:
                    # Lambda: read channels of interest from Parquet
                    (lambda channels_of_interest:
                        # Lambda: compute PSD and unpack outputs
                        (lambda psd_freqs:
                            # Lambda: check for None in PSD/freqs before proceeding
                            (
                                # Lambda: calculate band power and write results
                                (lambda power:
                                    (pl.DataFrame([
                                        {'channel': epochs.ch_names[ch_idx], 'band': band_name, 'power': power(psd_freqs[0][ch_idx], psd_freqs[1], band)}
                                        for ch_idx in range(len(epochs.ch_names)) for band_name, band in bands.items()
                                    ]).write_parquet(get_output_filename(input_fif)),
                                    print(f"[Nextflow] PSD analysis finished for input: {input_fif}. Output: {get_output_filename(input_fif)}")
                                    ) if callable(power) else print(f"[Nextflow] Lambda for band power is not callable for input: {input_fif}")
                                )(
                                    # Lambda: mean power in band
                                    lambda arr, f, band: float(np.mean(arr[(f >= band[0]) & (f <= band[1])]))
                                ) if psd_freqs[0] is not None and psd_freqs[1] is not None else print(f"[Nextflow] PSD analysis failed for input: {input_fif} (PSD or freqs is None)")
                            )
                        )(
                            # Lambda: ensure tuple output for PSD/freqs
                            (lambda ensure_tuple:
                                ensure_tuple(
                                    epochs.compute_psd(fmin=min(b[0] for b in bands.values()), fmax=max(b[1] for b in bands.values()), picks=channels_of_interest)
                                ) if hasattr(epochs, 'compute_psd')
                                else ensure_tuple(
                                    mne.time_frequency.psd_array_welch(
                                        epochs.get_data(), sfreq=epochs.info['sfreq'], fmin=min(b[0] for b in bands.values()), fmax=max(b[1] for b in bands.values())
                                    )
                                )
                            )(
                                # Lambda: robust unpacking for all PSD/freqs output types
                                lambda out: (
                                    (out.get_data(), out.freqs) if hasattr(out, 'get_data') and hasattr(out, 'freqs') else
                                    (out[0], out[1]) if isinstance(out, tuple) and len(out) == 2 else
                                    (out, None) if isinstance(out, np.ndarray) else
                                    (np.asarray(out), None) if out is not None else (None, None)
                                )
                            )
                        )
                    )(
                        # Lambda: get channel list from Parquet
                        (lambda: (lambda df: df['channel'].to_list())(pl.read_parquet(channels_parquet)) if channels_parquet else None)()
                    )
                )(
                    # Lambda: convert bands Parquet to dict
                    (lambda df: {r['band_name']:(r['fmin'],r['fmax']) for r in df.to_dicts()})(pl.read_parquet(bands_parquet))
                )
            )(mne.read_epochs(input_fif, preload=True))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_fif, bands_parquet = args[1], args[2]
            channels_parquet = args[3] if len(args) > 3 and args[3].endswith('.parquet') else None
            run(input_fif, bands_parquet, channels_parquet)
    except Exception as e:
        print(f"[Nextflow] PSD analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}"); sys.exit(1)
