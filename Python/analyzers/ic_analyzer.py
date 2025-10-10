import polars as pl, mne, sys, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python ic_analyzer.py <input_file>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_ic.parquet"
    
    # Format-aware loading for optimal scientific quality
    load_eeg_data = lambda input_file: (
        mne.io.read_raw_fif(input_file, preload=True, verbose=False) 
        if input_file.endswith('.fif') else
        (lambda df: mne.io.RawArray(
            df.select([c for c in df.columns if c not in ['time', 'sfreq', 'data_type']]).to_numpy().T, 
            mne.create_info([c for c in df.columns if c not in ['time', 'sfreq', 'data_type']], 
                          sfreq=df['sfreq'][0] if 'sfreq' in df.columns else 256)
        ))(pl.read_parquet(input_file))
    )
    
    run = lambda input_file: (
        print(f"[Nextflow] ICA analysis started for input: {input_file}") or (
            # Load EEG data with format-aware loading (FIF preferred for scientific quality)
            (lambda raw:
                    # Fit ICA and apply to data
                    (lambda ica:
                        (lambda cleaned_raw:
                            pl.concat([
                                # Cleaned EEG data for downstream analysis
                                pl.DataFrame({
                                    **{ch: cleaned_raw.get_data(picks=[ch])[0] for ch in cleaned_raw.ch_names},
                                    'sfreq': [cleaned_raw.info['sfreq']] * len(cleaned_raw.times),
                                    'times': cleaned_raw.times,
                                    'data_type': ['cleaned_eeg'] * len(cleaned_raw.times)
                                }),
                                # ICA results for plotting
                                pl.DataFrame([
                                    {
                                        'component': idx, 
                                        'explained_var': float(ica.pca_explained_variance_[idx]),
                                        'data_type': 'analysis_result',
                                        'plot_type': 'bar', 
                                        'x_scale': 'ordinal', 
                                        'y_scale': 'nominal',
                                        'x_data': f'IC{idx}', 
                                        'y_data': float(ica.pca_explained_variance_[idx]),
                                        'y_label': 'Explained Variance', 
                                        'plot_weight': 1
                                    }
                                    for idx in range(len(ica.pca_explained_variance_))
                                ])
                            ]).write_parquet(get_output_filename(input_file)) or
                            print(f"[Nextflow] ICA analysis finished for input: {input_file}")
                        )(ica.apply(raw.copy()))
                    )((lambda fitted_ica: fitted_ica)(mne.preprocessing.ICA(n_components=0.99, random_state=42).fit(raw)))
                )(load_eeg_data(input_file))))
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_file = args[1]
            run(input_file)
    except Exception as e:
        print(f"[Nextflow] ICA analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}")
        sys.exit(1)
