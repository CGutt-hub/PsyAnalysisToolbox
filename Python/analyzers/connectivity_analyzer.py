import polars as pl, mne, sys, os, warnings, numpy as np
from mne_connectivity import spectral_connectivity_epochs
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def analyze_connectivity(ip: str, method: str = 'coh') -> str:
    """
    Analyze functional connectivity between fNIRS channels.
    Outputs folder structure with plot-ready data.
    
    Args:
        ip: Input .fif file with preprocessed fNIRS data
        method: Connectivity method (coh, pli, wpli, etc.)
    
    Returns:
        Path to signal file
    """
    print(f"[ANALYZE] Connectivity analysis started: {ip}")
    if not ip.endswith('.fif'): 
        print("[ANALYZE] Error: Connectivity requires MNE .fif format")
        sys.exit(1)
    
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=30.0, overlap=15.0, verbose=False)
    conn = spectral_connectivity_epochs(epochs, method=method, fmin=0.01, fmax=0.1, verbose=False)
    conn_matrix = conn.get_data(output='dense').mean(axis=2)  # Average across frequencies
    
    # Extract upper triangle (avoid duplicates)
    n_ch = conn_matrix.shape[0]
    ch_names = raw.ch_names
    pairs, values = [], []
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            pairs.append(f"{ch_names[i]}-{ch_names[j]}")
            values.append(conn_matrix[i, j])
    
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_connectivity")
    os.makedirs(out_folder, exist_ok=True)
    
    # Single condition output (all connectivity pairs)
    out = pl.DataFrame({
        'condition': ['connectivity'],
        'x_data': [pairs],
        'y_data': [values],
        'y_var': [[0.0] * len(values)],  # No variance for single connectivity estimate
        'plot_type': ['bar'],
        'x_label': ['Channel Pairs'],
        'y_label': [f'{method.upper()} Connectivity'],
        'count': [len(pairs)]
    })
    
    out.write_parquet(os.path.join(out_folder, f"{base}_connectivity1.parquet"))
    print(f"[ANALYZE]   {len(pairs)} channel pairs analyzed (mean {method}: {np.mean(values):.3f})")
    
    # Write signal file
    signal_path = os.path.join(workspace_root, f"{base}_connectivity.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [1],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] Connectivity analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__': (lambda a: analyze_connectivity(a[1], a[2] if len(a) > 2 else 'coh') if len(a) >= 2 else (print('[ANALYZE] Usage: python connectivity_analyzer.py <input.fif> [method]'), sys.exit(1)))(sys.argv)