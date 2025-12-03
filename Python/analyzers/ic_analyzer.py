import polars as pl, mne, sys, os, numpy as np, warnings

# Suppress MNE naming convention warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def analyze_ica(ip: str, n_components: float = 0.99) -> str:
    """
    Perform ICA on EEG data. Outputs cleaned data as .fif and component variance plots in folder.
    
    Args:
        ip: Input .fif file with EEG data
        n_components: Number of components (0-1 for variance explained, >1 for exact number)
    
    Returns:
        Path to signal file
    """
    print(f"[ANALYZE] ICA analysis started: {ip}")
    
    if not ip.endswith('.fif'):
        raise ValueError("ICA analyzer requires MNE .fif input")
    
    # Load EEG data
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    print(f"[ANALYZE] Loaded: {len(raw.ch_names)} channels, {len(raw.times)} samples, sfreq={raw.info['sfreq']} Hz")
    
    # Fit ICA
    print(f"[ANALYZE] Fitting ICA (n_components={n_components})...")
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, verbose=False)
    ica.fit(raw)
    
    n_ics = ica.n_components_
    print(f"[ANALYZE] ICA fitted: {n_ics} independent components")
    
    # Apply ICA to get cleaned data
    cleaned_raw = ica.apply(raw.copy())
    
    # Create output folder
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_ica")
    os.makedirs(out_folder, exist_ok=True)
    
    # Save cleaned EEG data as .fif
    cleaned_fif = os.path.join(out_folder, f"{base}_ica_cleaned.fif")
    cleaned_raw.save(cleaned_fif, overwrite=True, verbose=False)
    print(f"[ANALYZE]   Cleaned EEG: {os.path.basename(cleaned_fif)}")
    
    # Create component variance plot data
    variance_data = pl.DataFrame({
        'x_data': [[f'IC{i}' for i in range(n_ics)]],
        'y_data': [ica.pca_explained_variance_[:n_ics].tolist()],
        'plot_type': ['bar'],
        'x_label': ['Independent Component'],
        'y_label': ['Explained Variance (%)']
    })
    
    variance_file = os.path.join(out_folder, f"{base}_ica1.parquet")
    variance_data.write_parquet(variance_file)
    print(f"[ANALYZE]   Component variance: {os.path.basename(variance_file)}")
    
    # Write signal file in workspace root
    signal_path = os.path.join(workspace_root, f"{base}_ica.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'n_components': [n_ics],
        'cleaned_fif': [cleaned_fif]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] ICA analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_ica(a[1], float(a[2]) if len(a) > 2 else 0.99) if len(a) >= 2 else (
        print("[ANALYZE] Usage: python ic_analyzer.py <input.fif> [n_components]"),
        sys.exit(1)
    ))(sys.argv)
