import mne, polars as pl, numpy as np, sys, os
if __name__ == "__main__":
    # Lambda: print usage and exit if arguments are missing
    usage = lambda: print("Usage: python eeg_preprocessor.py <input_file> <participant_id> [l_freq] [h_freq] [reference] [output_parquet]") or sys.exit(1)
    # Lambda: reconstruct MNE Raw object from Parquet tabular data
    parquet_to_raw = lambda df, sfreq, ch_names: mne.io.RawArray(
        np.array([df[ch].to_numpy() for ch in ch_names]),
        mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    )
    # Lambda: main preprocessing logic, auto-detects file type and applies steps
    run = lambda input_file, participant_id, l_freq, h_freq, reference, output_parquet: (
        print(f"[Nextflow] EEG preprocessing started for participant: {participant_id}") or (
            # Lambda: get file extension and branch logic
            (lambda ext: (
                # Lambda: FIF branch, reads and preprocesses MNE Raw from FIF
                (lambda raw: (
                    # Lambda: filter EEG data (in-place)
                    (lambda _: (
                        # Lambda: set EEG reference (in-place)
                        (lambda _: (
                            # Lambda: write processed EEG to Parquet (channels x samples)
                            pl.DataFrame({ch: raw.get_data(picks=[ch])[0] for ch in raw.ch_names} | {'participant_id': participant_id}).write_parquet(output_parquet),
                            print(f"[Nextflow] EEG preprocessing finished for participant: {participant_id}")
                        ))(raw.set_eeg_reference(reference, verbose=False) if hasattr(raw, 'set_eeg_reference') else None)
                    ))(raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False) if hasattr(raw, 'filter') else None)
                ) if not np.isnan(raw.get_data()).any() else (
                    # Lambda: handle NaNs in EEG data
                    print(f"[Nextflow] EEG preprocessing errored for participant: {participant_id}. NaNs detected in EEG data."),
                    pl.DataFrame([]).write_parquet(output_parquet),
                    sys.exit(1)
                ))(mne.io.read_raw_fif(input_file, preload=True)) if ext==".fif" else
                # Lambda: Parquet branch, reads tabular EEG and reconstructs MNE Raw
                (lambda df: (
                    # Lambda: extract channel names and sampling frequency from Parquet
                    (lambda ch_names, sfreq: (
                        # Lambda: reconstruct MNE Raw and preprocess
                        (lambda raw: (
                            # Lambda: set EEG reference (in-place)
                            (lambda _: (
                                # Lambda: write processed EEG to Parquet (channels x samples)
                                pl.DataFrame({ch: raw.get_data(picks=[ch])[0] for ch in ch_names} | {'participant_id': participant_id}).write_parquet(output_parquet),
                                print(f"[Nextflow] EEG preprocessing finished for participant: {participant_id}")
                            ))(raw.set_eeg_reference(reference, verbose=False) if hasattr(raw, 'set_eeg_reference') else None)
                        ) if not np.isnan(raw.get_data()).any() else (
                            # Lambda: handle NaNs in EEG data
                            print(f"[Nextflow] EEG preprocessing errored for participant: {participant_id}. NaNs detected in EEG data."),
                            pl.DataFrame([]).write_parquet(output_parquet),
                            sys.exit(1)
                        )
                        )(parquet_to_raw(df, sfreq, ch_names))
                    ))([c for c in df.columns if c not in ["participant_id", "sfreq"]], float(df.select("sfreq").to_numpy()[0]) if "sfreq" in df.columns else 250.0)
                ))(pl.read_parquet(input_file)) if ext==".parquet" else (
                    # Lambda: handle unsupported file types
                    print(f"[Nextflow] EEG preprocessing errored for participant: {participant_id}. Unsupported file type: {ext}"),
                    pl.DataFrame([]).write_parquet(output_parquet),
                    sys.exit(1)
                )
            ))(os.path.splitext(input_file)[1].lower())
        ) if l_freq is not None and h_freq is not None else (
            # Lambda: handle missing filter frequencies
            print(f"[Nextflow] EEG preprocessing errored for participant: {participant_id}. Missing filter frequencies."),
            pl.DataFrame([]).write_parquet(output_parquet),
            sys.exit(1)
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_file, participant_id = args[1], args[2]
            l_freq = float(args[3]) if len(args) > 3 else 1.0
            h_freq = float(args[4]) if len(args) > 4 else 40.0
            reference = args[5] if len(args) > 5 else 'average'
            output_parquet = args[6] if len(args) > 6 else f"{participant_id}_eeg.parquet"
            run(input_file, participant_id, l_freq, h_freq, reference, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] EEG preprocessing errored for participant: {pid}. Error: {e}")
        sys.exit(1)