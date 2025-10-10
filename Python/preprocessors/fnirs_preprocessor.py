import mne, polars as pl, numpy as np, sys, os
from mne_nirs.signal_enhancement import short_channel_regression
if __name__ == "__main__":
    usage = lambda: print("Usage: python fnirs_preprocessor.py <input_file> [l_freq] [h_freq] [short_reg]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_fnirs.parquet"
    run = lambda input_file, l_freq, h_freq, short_reg: (
        print(f"[Nextflow] fNIRS preprocessing started for file: {input_file}") or
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
                                    print(f"[Nextflow] fNIRS preprocessing finished for file: {input_file} (FIF + parquet)")
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
                                or print(f"[Nextflow] fNIRS preprocessing finished for file: {input_file}")
                            )
                            if isinstance(raw, mne.io.BaseRaw) else print(f"[Nextflow] fNIRS preprocessing errored for file: {input_file}. Invalid raw object.")
                        )
                    )(short_channel_regression(mne.io.read_raw_fif(input_file, preload=True).filter(l_freq=l_freq, h_freq=h_freq, verbose=False)) if short_reg else mne.io.read_raw_fif(input_file, preload=True).filter(l_freq=l_freq, h_freq=h_freq, verbose=False)) if ext == ".fif" else
                    (lambda df: (
                        (lambda ch_names, sfreq: (
                            (lambda raw:
                                (
                                    (pl.from_pandas(raw.to_data_frame()).write_parquet(get_output_filename(input_file))
                                     or print(f"[Nextflow] fNIRS preprocessing finished for file: {input_file}"))
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
                                        or print(f"[Nextflow] fNIRS preprocessing finished for file: {input_file}")
                                    )
                                    if isinstance(raw, mne.io.BaseRaw) else print(f"[Nextflow] fNIRS preprocessing errored for file: {input_file}. Invalid raw object.")
                                )
                            )(short_channel_regression(parquet_to_raw(df, sfreq, ch_names)) if short_reg else parquet_to_raw(df, sfreq, ch_names))
                        ))([c for c in df.columns if c != "sfreq"], float(df.select("sfreq").to_numpy()[0]) if "sfreq" in df.columns else 10.0)
                    ))(pl.read_parquet(input_file)) if ext == ".parquet" else (
                        print(f"[Nextflow] fNIRS preprocessing errored for file: {input_file}. Unsupported file type: {ext}") or
                        pl.DataFrame([]).write_parquet(get_output_filename(input_file)) or
                        sys.exit(1)
                    )
                ))(lambda df, sfreq, ch_names: mne.io.RawArray(
                    np.array([df[ch].to_numpy() for ch in ch_names]),
                    mne.create_info(list(ch_names), sfreq, ch_types="fnirs")
                ))
            ))(os.path.splitext(input_file)[1].lower())
        ) if l_freq is not None and h_freq is not None else (
            print(f"[Nextflow] fNIRS preprocessing errored for file: {input_file}. Missing filter frequencies.") or
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
            run(input_file, l_freq, h_freq, short_reg)
    except Exception as e:
        print(f"[Nextflow] fNIRS preprocessing errored for file: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)