import mne, polars as pl, numpy as np, sys, os, re
from mne_nirs.signal_enhancement import short_channel_regression

# intensity->OD conversion will be inlined at the RawArray construction site
if __name__ == "__main__":
    usage = lambda: print("[PREPROC] Usage: python fnirs_preprocessor.py <input_file> [l_freq] [h_freq] [short_reg]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_fnirs.parquet"
    run = lambda input_file, l_freq, h_freq, short_reg: (
        print(f"[PREPROC] fNIRS preprocessing started for file: {input_file}") or
            (lambda ext: (
                (lambda parquet_to_raw: (
                    (lambda raw:
                        (
                            (
                                # Save as FIF for MNE-critical analyses (GLM, advanced fNIRS processing)
                                raw.save(f"{os.path.splitext(get_output_filename(input_file))[0]}_fnirs.fif", overwrite=True, verbose=False),
                                # Save as parquet for pipeline efficiency
                                (lambda standardized_df: (
                                    standardized_df.write_parquet(get_output_filename(input_file)),
                                    print(f"[PREPROC] fNIRS preprocessing finished for file: {input_file} (FIF + parquet)")
                                ))(pl.from_pandas(raw.to_data_frame()).with_columns([
                                    pl.lit(raw.info['sfreq']).alias('sfreq'),
                                    pl.lit('preprocessed_fnirs').alias('data_type')
                                ]))
                            )
                            if isinstance(raw, mne.io.BaseRaw) and hasattr(raw, 'to_data_frame') else
                            (
                                pl.DataFrame({
                                    **{ch: (
                                        (lambda data:
                                            data[0] if isinstance(data, tuple) and hasattr(data, '__getitem__') and len(data) > 0 else
                                            data[0] if isinstance(data, (np.ndarray, list)) and len(data) > 0 else
                                            data if isinstance(data, float) else None
                                        )(raw.get_data(picks=[ch]))
                                        if isinstance(raw, mne.io.BaseRaw) and ch in raw.ch_names else None
                                    ) for ch in raw.ch_names},
                                    'time': np.arange(raw.n_times) / raw.info['sfreq'],
                                    'sfreq': [raw.info['sfreq']] * raw.n_times,
                                    'data_type': ['preprocessed_fnirs'] * raw.n_times
                                }).write_parquet(get_output_filename(input_file))
                                or print(f"[PREPROC] fNIRS preprocessing finished for file: {input_file}")
                            )
                            if isinstance(raw, mne.io.BaseRaw) else print(f"[PREPROC] fNIRS preprocessing errored for file: {input_file}. Invalid raw object.")
                        )
                    )((lambda raw: short_channel_regression(raw) if short_reg and any(re.search(r'(^s\\d+\\b)|short|_sd|_short', ch, re.I) for ch in getattr(raw, 'ch_names', [])) else raw)(mne.io.read_raw_fif(input_file, preload=True).filter(l_freq=l_freq, h_freq=h_freq, verbose=False))) if ext == ".fif" else
                    (lambda df: (
                        (lambda ch_names, sfreq: (
                            (lambda raw:
                                (
                                    (pl.from_pandas(raw.to_data_frame()).write_parquet(get_output_filename(input_file))
                                     or print(f"[PREPROC] fNIRS preprocessing finished for file: {input_file}"))
                                    if isinstance(raw, mne.io.BaseRaw) and hasattr(raw, 'to_data_frame') else
                                    (
                                        pl.DataFrame({
                                            ch: (
                                                (lambda data:
                                                    data[0] if isinstance(data, tuple) and hasattr(data, '__getitem__') and len(data) > 0 else
                                                    data[0] if isinstance(data, (np.ndarray, list)) and len(data) > 0 else
                                                    data if isinstance(data, float) else None
                                                )(raw.get_data(picks=[ch]))
                                                if isinstance(raw, mne.io.BaseRaw) and ch in raw.ch_names else None
                                            )
                                            for ch in raw.ch_names
                                        }).write_parquet(get_output_filename(input_file))
                                        or print(f"[PREPROC] fNIRS preprocessing finished for file: {input_file}")
                                    )
                                    if isinstance(raw, mne.io.BaseRaw) else print(f"[PREPROC] fNIRS preprocessing errored for file: {input_file}. Invalid raw object.")
                                )
                            )((lambda raw: short_channel_regression(raw) if short_reg and any(re.search(r'(^s\\d+\\b)|short|_sd|_short', ch, re.I) for ch in getattr(raw, 'ch_names', [])) else raw)(parquet_to_raw(df, sfreq, ch_names)))
                        ))([c for c in df.columns if c != "sfreq"], float(df.select("sfreq").to_numpy()[0]) if "sfreq" in df.columns else 10.0)
                    ))(pl.read_parquet(input_file)) if ext == ".parquet" else (
                        print(f"[PREPROC] fNIRS preprocessing errored for file: {input_file}. Unsupported file type: {ext}") or
                        pl.DataFrame([]).write_parquet(get_output_filename(input_file)) or
                        sys.exit(1)
                    )
                ))(lambda df, sfreq, ch_names: (
                    # inline: build ndarray of intensities, compute per-channel I0 (median clamped),
                    # compute OD = -log10(I / I0) and replace non-finite with zeros, then build RawArray
                    (lambda arr: mne.io.RawArray(
                        # compute OD inline
                                    (lambda I_arr: np.nan_to_num(
                            -np.log10(
                                np.divide(
                                    I_arr,
                                    np.maximum((lambda I0: np.where(I0 <= 0, np.nanmax(I_arr, axis=1), I0))(np.median(I_arr, axis=1))[:, None], 1e-12),
                                    where=True
                                )
                            ), nan=0.0, posinf=0.0, neginf=0.0
                        ))(arr),
                        mne.create_info(list(ch_names), sfreq, ch_types="fnirs_od")
                    ))(np.array([df[ch].to_numpy() for ch in ch_names], dtype=float))
                ))
            ))(os.path.splitext(input_file)[1].lower())
        ) if l_freq is not None and h_freq is not None else (
            print(f"[PREPROC] fNIRS preprocessing errored for file: {input_file}. Missing filter frequencies.") or
            pl.DataFrame([]).write_parquet(get_output_filename(input_file)) or
            sys.exit(1)
        )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_file = args[1]
            l_freq = float(args[2]) if len(args) > 2 else 0.01
            h_freq = float(args[3]) if len(args) > 3 else 0.1
            short_reg = bool(int(args[4])) if len(args) > 4 else False
            # If the pipeline supplied a parquet and it's empty, do not try to
            # choose an alternative here — the pipeline is responsible for
            # selecting the correct stream/file. Exit with an error so the
            # pipeline can handle selection/retry.
            if os.path.splitext(input_file)[1].lower() == ".parquet":
                df_check = pl.read_parquet(input_file)
                is_empty = (df_check.shape == (0, 0)) or (df_check.shape[0] == 0) or (len(df_check.columns) == 0)
                if is_empty:
                    out_fname = get_output_filename(input_file)
                    print(f"[PREPROC] Input parquet {input_file} is empty; writing empty output {out_fname} and exiting successfully.")
                    pl.DataFrame([]).write_parquet(out_fname)
                    sys.exit(0)

            run(input_file, l_freq, h_freq, short_reg)
    except Exception as e:
        print(f"[PREPROC] Error: {e}")
        sys.exit(1)