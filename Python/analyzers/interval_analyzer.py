import polars as pl, numpy as np, sys, os

def analyze_intervals(ip: str, event_col: str | None = None, y_lim: float | None = None, 
                      y_label: str = 'Value (ms)', suffix: str = 'interval',
                      metrics_mode: str = 'auto') -> str:
    """
    Analyze inter-event intervals (SDNN, RMSSD) per condition from epoched point process data.
    Generic interval analyzer - works on any point process (R-peaks, blinks, keystrokes, etc.)
    
    Args:
        ip: Input parquet with epoched event data [condition, epoch_id, event_col, sfreq]
        event_col: Column containing event samples/times (auto-detected if None)
        y_lim: Optional Y-axis maximum limit
        y_label: Label for y-axis (e.g., 'Value (ms)', 'IBI (ms)')
        suffix: Output file suffix (default 'interval', use 'hrv' for HRV compatibility)
        metrics_mode: 'auto' (both SDNN+RMSSD), 'SDNN', or 'RMSSD' for single metric
    
    Returns:
        Path to signal file
    """
    print(f"[interval] Interval analysis: {ip}")
    df = pl.read_parquet(ip)
    
    if 'condition' not in df.columns or 'epoch_id' not in df.columns:
        raise ValueError("Input must have 'condition' and 'epoch_id' columns")
    
    # Auto-detect event column
    if event_col is None:
        candidates = ['R_Peak_Sample', 'rpeaks', 'peaks', 'events', 'samples']
        event_col = next((c for c in candidates if c in df.columns), None)
        if event_col is None:
            # Use first non-meta column
            meta_cols = ['time', 'sfreq', 'epoch_id', 'condition']
            event_col = [c for c in df.columns if c not in meta_cols][0]
    
    print(f"[interval] Using event column: {event_col}")
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else 1000.0
    
    base = os.path.splitext(os.path.basename(ip))[0]
    out_folder = os.path.join(os.getcwd(), f"{base}_{suffix}")
    os.makedirs(out_folder, exist_ok=True)
    
    conditions = sorted(df['condition'].unique().to_list())
    print(f"[interval] Processing {len(conditions)} conditions (sfreq={sfreq} Hz)")
    
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        epoch_ids = cond_df['epoch_id'].unique().to_list()
        
        sdnn_per_epoch, rmssd_per_epoch = [], []
        
        for eid in epoch_ids:
            epoch_df = cond_df.filter(pl.col('epoch_id') == eid)
            events = epoch_df[event_col].to_numpy()
            
            if len(events) < 2:
                continue
            
            # Calculate inter-event intervals in milliseconds
            intervals = np.diff(events) / sfreq * 1000.0
            
            if len(intervals) < 2:
                continue
            
            # SDNN: Standard deviation of intervals
            sdnn_per_epoch.append(float(np.std(intervals, ddof=1)))
            
            # RMSSD: Root mean square of successive differences
            rmssd_per_epoch.append(float(np.sqrt(np.mean(np.diff(intervals) ** 2))))
        
        if not sdnn_per_epoch:
            print(f"[interval] Warning: {cond} has no valid epochs, skipping")
            continue
        
        # Calculate mean and SEM across epochs
        sdnn_mean = float(np.mean(sdnn_per_epoch))
        sdnn_sem = float(np.std(sdnn_per_epoch, ddof=1) / np.sqrt(len(sdnn_per_epoch))) if len(sdnn_per_epoch) > 1 else 0.0
        rmssd_mean = float(np.mean(rmssd_per_epoch))
        rmssd_sem = float(np.std(rmssd_per_epoch, ddof=1) / np.sqrt(len(rmssd_per_epoch))) if len(rmssd_per_epoch) > 1 else 0.0
        
        # Build output based on metrics_mode
        if metrics_mode.upper() == 'SDNN':
            x_data, y_data, y_var = ['SDNN'], [sdnn_mean], [sdnn_sem]
        elif metrics_mode.upper() == 'RMSSD':
            x_data, y_data, y_var = ['RMSSD'], [rmssd_mean], [rmssd_sem]
        else:  # auto - both metrics
            x_data, y_data, y_var = ['SDNN', 'RMSSD'], [sdnn_mean, rmssd_mean], [sdnn_sem, rmssd_sem]
        
        output = pl.DataFrame({
            'condition': [str(cond)],
            'x_data': [x_data],
            'y_data': [y_data],
            'y_var': [y_var],
            'plot_type': ['bar'],
            'x_label': ['Interval Metric'],
            'y_label': [y_label],
            'y_ticks': [y_lim] if y_lim is not None else [None]
        })
        
        out_path = os.path.join(out_folder, f"{base}_{suffix}{idx+1}.parquet")
        output.write_parquet(out_path)
        print(f"[interval]   {cond}: SDNN={sdnn_mean:.2f}±{sdnn_sem:.2f}, RMSSD={rmssd_mean:.2f}±{rmssd_sem:.2f} ({len(sdnn_per_epoch)} epochs)")
    
    signal_path = os.path.join(os.getcwd(), f"{base}_{suffix}.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[interval] Output: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_intervals(a[1],
                                  a[2] if len(a) > 2 and a[2] and a[2] != 'None' else None,
                                  float(a[3]) if len(a) > 3 and a[3] and a[3] != 'None' else None,
                                  a[4] if len(a) > 4 else 'Value (ms)',
                                  a[5] if len(a) > 5 else 'interval',
                                  a[6] if len(a) > 6 else 'auto') if len(a) >= 2 else (
        print('Compute interval statistics (SDNN, RMSSD). Plot-ready output.'),
        print('[interval] Usage: python interval_analyzer.py <epochs.parquet> [event_col] [y_lim] [y_label] [suffix] [metrics_mode]'),
        print('[interval] metrics_mode: auto (both SDNN+RMSSD), SDNN, or RMSSD'),
        print('[interval] Example: python interval_analyzer.py ecg_epochs.parquet R_Peak_Sample None "IBI (ms)" hrv auto'),
        sys.exit(1)))(sys.argv)
