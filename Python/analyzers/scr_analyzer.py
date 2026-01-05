import polars as pl, sys, os

def analyze_scr(ip: str) -> str:
    """
    Analyze SCR data from epoched EDA. Outputs one file per condition in folder structure.
    
    Args:
        ip: Input parquet file with epoched EDA data (flat format with condition/epoch_id)
    
    Returns:
        Path to signal file
    """
    print(f"[ANALYZE] SCR analysis started: {ip}")
    df = pl.read_parquet(ip)
    
    if 'condition' not in df.columns or 'epoch_id' not in df.columns:
        raise ValueError("Input must have 'condition' and 'epoch_id' columns (flat epoched format)")
    
    eda_col = [c for c in df.columns if c not in ['time', 'sfreq', 'epoch_id', 'condition']][0]
    conditions = sorted(df['condition'].unique().to_list())
    
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_scr")
    os.makedirs(out_folder, exist_ok=True)
    
    print(f"[ANALYZE] Processing {len(conditions)} conditions: {conditions}")
    
    # Process each condition separately
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        
        # Compute epoch-averaged waveforms with relative time
        cond_df = cond_df.with_columns([
            (pl.col('time') - pl.col('time').min().over('epoch_id')).alias('relative_time')
        ])
        
        # Average across epochs with SEM, then downsample (every 50th point for manageable file size)
        result = cond_df.group_by('relative_time').agg([
            pl.col(eda_col).mean().alias('mean_eda'),
            (pl.col(eda_col).std() / pl.col(eda_col).count().sqrt()).alias('sem_eda')
        ]).sort('relative_time').with_row_index().filter(
            pl.col('index') % 50 == 0
        ).drop('index')
        
        # Output format for plotter with error bars (line_grid for grid layout)
        output = pl.DataFrame({
            'condition': [cond],
            'x_data': [result['relative_time'].to_list()],
            'y_data': [result['mean_eda'].to_list()],
            'y_var': [result['sem_eda'].to_list()],
            'plot_type': ['line_grid'],
            'x_label': ['Time from onset (s)'],
            'y_label': ['Mean EDA (\u03bcS)']
        })
        
        out_path = os.path.join(out_folder, f"{base}_scr{idx+1}.parquet")
        output.write_parquet(out_path)
        print(f"[ANALYZE]   {cond}: {os.path.basename(out_path)} ({len(result)} points)")
    
    # Write signal file in workspace root
    signal_path = os.path.join(workspace_root, f"{base}_scr.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[ANALYZE] SCR analysis finished. Signal: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_scr(a[1]) if len(a) >= 2 else (
        print("[ANALYZE] Usage: python scr_analyzer.py <input_epochs.parquet>"),
        sys.exit(1)
    ))(sys.argv)
