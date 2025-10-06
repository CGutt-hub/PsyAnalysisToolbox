import polars as pl, mne, sys
if __name__ == "__main__":


    import os
    usage = lambda: print("Usage: python ic_analyzer.py <input_parquet>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_ic.parquet"
    run = lambda input_parquet: (
        print(f"[Nextflow] ICA analysis started for input: {input_parquet}") or (
            # Lambda: read EEG data from Parquet and convert to MNE RawArray
            (lambda df:
                (lambda raw:
                    # Lambda: fit ICA and extract components
                    (lambda ica:
                        # Lambda: collect ICA results and write to Parquet
                        (pl.DataFrame([
                            {'component': idx, 'explained_var': var}
                            for idx, var in enumerate(ica.get_explained_variance_ratio(raw).values())
                        ]).write_parquet(get_output_filename(input_parquet)),
                         print(f"[Nextflow] ICA analysis finished for input: {input_parquet}"))
                    )(mne.preprocessing.ICA(n_components=0.99, random_state=42).fit(raw))
                )(mne.io.RawArray(df.select([c for c in df.columns if c != 'time']).to_numpy().T, mne.create_info([c for c in df.columns if c != 'time'], sfreq=256)))
            )(pl.read_parquet(input_parquet))))
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_parquet = args[1]
            run(input_parquet)
    except Exception as e:
        print(f"[Nextflow] ICA analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}")
        sys.exit(1)
