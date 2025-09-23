import polars as pl, mne, sys
if __name__ == "__main__":
    usage = lambda: print("Usage: python ic_analysis.py <input_parquet> <participant_id> [output_parquet]") or sys.exit(1)
    run = lambda input_parquet, participant_id, output_parquet: (
        print(f"[Nextflow] ICA analysis started for participant: {participant_id}") or (
            # Lambda: read EEG data from Parquet and convert to MNE RawArray
            (lambda df:
                (lambda raw:
                    # Lambda: fit ICA and extract components
                    (lambda ica:
                        # Lambda: collect ICA results and write to Parquet
                        (pl.DataFrame([
                            {'component': idx, 'explained_var': var, 'participant_id': participant_id}
                            for idx, var in enumerate(ica.get_explained_variance_ratio(raw).values())
                        ]).write_parquet(output_parquet),
                         print(f"[Nextflow] ICA analysis finished for participant: {participant_id}"))
                    )(mne.preprocessing.ICA(n_components=0.99, random_state=42).fit(raw))
                )(mne.io.RawArray(df.select([c for c in df.columns if c != 'time']).to_numpy().T, mne.create_info([c for c in df.columns if c != 'time'], sfreq=256)))
            )(pl.read_parquet(input_parquet))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet, participant_id = args[1], args[2]
            output_parquet = args[3] if len(args) > 3 else f"{participant_id}_ica.parquet"
            run(input_parquet, participant_id, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] ICA analysis errored for participant: {pid}. Error: {e}")
        sys.exit(1)
