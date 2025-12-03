import polars as pl, numpy as np, sys, os

def analyze_hrv(ip: str) -> str:
    """Analyze HRV per condition from flat epoched R-peaks data. Outputs folder structure with signal file."""
    print(f"[ANALYZE] HRV analysis started: {ip}")
    df = pl.read_parquet(ip)
    
    if 'condition' not in df.columns or 'epoch_id' not in df.columns:
        raise ValueError("Input must have 'condition' and 'epoch_id' columns (flat epoched format)")
    
    rpeak_col = 'R_Peak_Sample' if 'R_Peak_Sample' in df.columns else 'rpeaks'
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else 1000.0
    
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_hrv")
    os.makedirs(out_folder, exist_ok=True)
    
    conditions = sorted(df['condition'].unique().to_list())
    print(f"[ANALYZE] Processing {len(conditions)} conditions (sfreq={sfreq} Hz)")
    
    # Process each condition separately - one file per condition
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        
        # Group by epoch to calculate per-epoch HRV, then compute mean and SEM across epochs
        epoch_ids = cond_df['epoch_id'].unique().to_list()
        sdnn_per_epoch = []
        rmssd_per_epoch = []
        
        for eid in epoch_ids:
            epoch_df = cond_df.filter(pl.col('epoch_id') == eid)
            rpeaks = epoch_df[rpeak_col].to_numpy()
            
            if len(rpeaks) < 2:
                continue
            
            # Calculate RR intervals in milliseconds
            rr_intervals = np.diff(rpeaks) / sfreq * 1000.0
            
            if len(rr_intervals) < 2:
                continue
            
            # SDNN: Standard deviation of NN intervals
            sdnn_per_epoch.append(float(np.std(rr_intervals, ddof=1)))
            
            # RMSSD: Root mean square of successive differences
            rmssd_per_epoch.append(float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))))
        
        if not sdnn_per_epoch:
            print(f"[ANALYZE]   {cond}: No valid epochs, skipping")
            continue
        
        # Calculate mean and SEM across epochs
        sdnn_mean = float(np.mean(sdnn_per_epoch))
        sdnn_sem = float(np.std(sdnn_per_epoch, ddof=1) / np.sqrt(len(sdnn_per_epoch)))
        rmssd_mean = float(np.mean(rmssd_per_epoch))
        rmssd_sem = float(np.std(rmssd_per_epoch, ddof=1) / np.sqrt(len(rmssd_per_epoch)))
        
        # Output format for plotter: bar chart with HRV metrics and error bars
        output = pl.DataFrame({
            'condition': [cond],
            'x_data': [['SDNN', 'RMSSD']],
            'y_data': [[sdnn_mean, rmssd_mean]],
            'y_var': [[sdnn_sem, rmssd_sem]],
            'plot_type': ['bar'],
            'x_label': ['HRV Metric'],
            'y_label': ['Value (ms)']
        })
        
        out_path = os.path.join(out_folder, f"{base}_hrv{idx+1}.parquet")
        output.write_parquet(out_path)
        print(f"[ANALYZE]   {cond}: SDNN={sdnn_mean:.2f}±{sdnn_sem:.2f} ms, RMSSD={rmssd_mean:.2f}±{rmssd_sem:.2f} ms ({len(sdnn_per_epoch)} epochs) -> {os.path.basename(out_path)}")
    
    # Write signal file in workspace root
    signal_path = os.path.join(workspace_root, f"{base}_hrv.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] HRV analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_hrv(a[1]) if len(a) >= 2 else (
        print("[ANALYZE] Usage: python hrv_analyzer.py <input_epochs.parquet>"),
        sys.exit(1)
    ))(sys.argv)
