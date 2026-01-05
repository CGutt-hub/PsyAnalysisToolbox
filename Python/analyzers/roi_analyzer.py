import polars as pl, mne, sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def analyze_roi(ip: str, roi_config: str) -> str:
    """
    Analyze fNIRS data aggregated by brain regions (ROIs).
    Outputs folder structure with plot-ready data.
    
    Args:
        ip: Input .fif file with preprocessed fNIRS data
        roi_config: JSON string or path defining ROI channel groups
    
    Returns:
        Path to signal file
    """
    print(f"[ANALYZE] ROI analysis started: {ip}")
    if not ip.endswith('.fif'): 
        print("[ANALYZE] Error: ROI analysis requires MNE .fif format")
        sys.exit(1)
    
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    rois = json.loads(roi_config) if not roi_config.endswith('.json') else json.load(open(roi_config))
    
    roi_names, roi_means, roi_stds = [], [], []
    for name, indices in rois.items():
        valid_indices = [i for i in indices if i < len(raw.ch_names)]
        if not valid_indices:
            print(f"[ANALYZE] Warning: ROI '{name}' has no valid channels, skipping")
            continue
        roi_data = raw.get_data([raw.ch_names[i] for i in valid_indices])
        roi_mean = np.mean(roi_data)  # Mean across all channels and time
        roi_std = np.std(roi_data)
        roi_names.append(name)
        roi_means.append(float(roi_mean))
        roi_stds.append(float(roi_std))
    
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_roi")
    os.makedirs(out_folder, exist_ok=True)
    
    # Single condition output (all ROIs)
    out = pl.DataFrame({
        'condition': ['roi_activity'],
        'x_data': [roi_names],
        'y_data': [roi_means],
        'y_var': [roi_stds],
        'plot_type': ['bar'],
        'x_label': ['Brain Region'],
        'y_label': ['Mean Activity (a.u.)'],
        'count': [len(roi_names)]
    })
    
    out.write_parquet(os.path.join(out_folder, f"{base}_roi1.parquet"))
    print(f"[ANALYZE]   {len(roi_names)} ROIs analyzed: {roi_names}")
    
    # Write signal file
    signal_path = os.path.join(workspace_root, f"{base}_roi.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [1],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] ROI analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__': (lambda a: analyze_roi(a[1], a[2]) if len(a) >= 3 else (print('[ANALYZE] Usage: python roi_analyzer.py <input.fif> <roi_config_json>'), sys.exit(1)))(sys.argv)
