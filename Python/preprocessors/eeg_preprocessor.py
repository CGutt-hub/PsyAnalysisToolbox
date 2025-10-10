import mne, polars as pl, numpy as np, sys, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python eeg_preprocessor.py <input_file> [l_freq] [h_freq] [reference]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_eeg.parquet"
    run = lambda input_file, l_freq, h_freq, reference: (
        # Lambda: reconstruct MNE Raw object from Parquet tabular data (nested)
        (lambda parquet_to_raw: (
        print(f"[Nextflow] EEG preprocessing started for participant: {input_file}") or (
            (lambda ext: (
                (lambda raw: (
                    (lambda _: (
                        (lambda _: (
                            pl.DataFrame({ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names}).write_parquet(get_output_filename(input_file)),
                            print(f"[Nextflow] EEG preprocessing finished for file: {input_file}")
                        ))(raw.set_eeg_reference(reference, verbose=False) if hasattr(raw, 'set_eeg_reference') else None)
                    ))(raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False) if hasattr(raw, 'filter') else None)
                ) if not np.isnan(raw.get_data()).any() else (
                    print(f"[Nextflow] EEG preprocessing errored for file: {input_file}. NaNs detected in EEG data."),
                    pl.DataFrame([]).write_parquet(get_output_filename(input_file)),
                    sys.exit(1)
                ))(mne.io.read_raw_fif(input_file, preload=True)) if ext==".fif" else
                (lambda df: (
                    (lambda ch_names, sfreq: (
                        (lambda raw: (
                            (lambda _: (
                                # Save as FIF for MNE-critical analyses (IC, advanced EEG processing)
                                raw.save(f"{os.path.splitext(get_output_filename(input_file))[0]}_eeg.fif", overwrite=True, verbose=False),
                                # Save as parquet for pipeline efficiency (epoching, merging, etc.)
                                (lambda standardized_df: (
                                    standardized_df.write_parquet(get_output_filename(input_file)),
                                    print(f"[Nextflow] EEG preprocessing finished for file: {input_file} (FIF + parquet)")
                                ))(pl.DataFrame({
                                    **{ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names},
                                    'time': np.arange(raw.n_times) / raw.info['sfreq'],
                                    'sfreq': [raw.info['sfreq']] * raw.n_times,
                                    'data_type': ['preprocessed_eeg'] * raw.n_times
                                }))
                            ))(raw.set_eeg_reference(reference, verbose=False) if hasattr(raw, 'set_eeg_reference') else None)
                        ) if not np.isnan(raw.get_data()).any() else (
                            print(f"[Nextflow] EEG preprocessing errored for file: {input_file}. NaNs detected in EEG data."),
                            pl.DataFrame([]).write_parquet(get_output_filename(input_file)),
                            sys.exit(1)
                        )
                        )(parquet_to_raw(df, sfreq, ch_names))
                    ))([c for c in df.columns if c != "sfreq"], float(df.select("sfreq").to_numpy()[0]) if "sfreq" in df.columns else 250.0)
                ))(pl.read_parquet(input_file)) if ext==".parquet" else (
                    print(f"[Nextflow] EEG preprocessing errored for file: {input_file}. Unsupported file type: {ext}"),
                    pl.DataFrame([]).write_parquet(get_output_filename(input_file)),
                    sys.exit(1)
                )
            ))(os.path.splitext(input_file)[1].lower())
        ) if l_freq is not None and h_freq is not None else (
            # Lambda: handle missing filter frequencies
            print(f"[Nextflow] EEG preprocessing errored for file: {input_file}. Missing filter frequencies."),
            pl.DataFrame([]).write_parquet(get_output_filename(input_file)),
            sys.exit(1)
        )
        ) # end parquet_to_raw lambda
    )(lambda df, sfreq, ch_names: mne.io.RawArray(
        np.array([df[ch].to_numpy() for ch in ch_names]),
        mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    ))
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_file = args[1]
            l_freq = float(args[2]) if len(args) > 2 else 1.0
            h_freq = float(args[3]) if len(args) > 3 else 40.0
            reference = args[4] if len(args) > 4 else 'average'
            run(input_file, l_freq, h_freq, reference)
    except Exception as e:
        print(f"[Nextflow] EEG preprocessing errored for file: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)