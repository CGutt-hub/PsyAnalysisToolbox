import polars as pl, sys, os, numpy as np

def analyze_eda(ip: str) -> str:
    """
    Analyze EDA per condition: mean conductance change (ΔμS) per trial.
    Outputs folder structure with signalling file.
    
    Args:
        ip: Input parquet with epoched EDA (flat format: condition, epoch_id, time, eda)
    
    Returns:
        Path to signal file
    """
    print(f"[ANALYZE] EDA analysis started: {ip}")
    df = pl.read_parquet(ip)
    
    if 'condition' not in df.columns or 'epoch_id' not in df.columns:
        raise ValueError("Input must have 'condition' and 'epoch_id' columns")
    
    eda_col = [c for c in df.columns if c not in ['time', 'sfreq', 'epoch_id', 'condition']][0]
    conditions = sorted(df['condition'].unique().to_list())
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_eda")
    os.makedirs(out_folder, exist_ok=True)
    
    print(f"[ANALYZE] Processing {len(conditions)} conditions: {conditions}")
    
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        epochs = cond_df['epoch_id'].unique().to_list()
        
        # Compute conductance change per epoch (max - baseline)
        changes = []
        for eid in epochs:
            epoch_data = cond_df.filter(pl.col('epoch_id') == eid)[eda_col].to_numpy()
            baseline = np.mean(epoch_data[:int(len(epoch_data)*0.2)])  # First 20% as baseline
            peak = np.max(epoch_data)
            changes.append(peak - baseline)
        
        # Output: mean change with SEM
        mean_change = float(np.mean(changes))
        sem_change = float(np.std(changes, ddof=1) / np.sqrt(len(changes))) if len(changes) > 1 else 0.0
        
        out = pl.DataFrame({
            'condition': [cond],
            'x_data': [['Mean ΔμS']],
            'y_data': [[mean_change]],
            'y_var': [[sem_change]],
            'plot_type': ['bar'],
            'x_label': [''],
            'y_label': ['Conductance Change (μS)'],
            'count': [len(changes)]
        })
        
        out.write_parquet(os.path.join(out_folder, f"{base}_eda{idx+1}.parquet"))
        print(f"[ANALYZE]   {cond}: ΔμS = {mean_change:.3f} ± {sem_change:.3f} ({len(changes)} trials)")
    
    # Write signal file
    signal_path = os.path.join(workspace_root, f"{base}_eda.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] EDA analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_eda(a[1]) if len(a) >= 2 else (
        print('[ANALYZE] Usage: python eda_analyzer.py <epochs.parquet>'), sys.exit(1)))(sys.argv)
