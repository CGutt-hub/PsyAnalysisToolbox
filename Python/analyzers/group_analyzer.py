import polars as pl, numpy as np, sys, os, json

def analyze_groups(ip: str, groups_config: str, y_lim: float | None = None, 
                   x_label: str = 'ROI', y_label: str = 'Mean', suffix: str = 'roi') -> str:
    """
    Aggregate channels by groups (ROIs) and compute group-level statistics per condition.
    Generic group analyzer - works on epoched multichannel data.
    
    Args:
        ip: Input parquet with epoched data (condition, epoch_id, time, channel_cols...)
        groups_config: JSON string defining groups, e.g. {"DLPFC": ["S1_D1", "S2_D1"], "VMPFC": ["S3_D2"]}
        y_lim: Optional Y-axis maximum limit
        x_label: Label for x-axis (e.g., 'ROI', 'Brain Region')
        y_label: Label for y-axis (e.g., 'HbO2 Change (μM)', 'Mean Activity')
        suffix: Output file suffix (default 'roi')
    
    Returns:
        Path to signal file
    """
    print(f"[group] ROI analysis: {ip}")
    
    # Parse groups config
    if os.path.isfile(groups_config):
        with open(groups_config) as f:
            groups = json.load(f)
    else:
        groups = json.loads(groups_config)
    
    df = pl.read_parquet(ip)
    meta_cols = ['time', 'sfreq', 'epoch_id', 'condition']
    ch_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"[group] Available channels ({len(ch_cols)}): {ch_cols[:10]}..." if len(ch_cols) > 10 else f"[group] Available channels: {ch_cols}")
    print(f"[group] Looking for ROIs: {list(groups.keys())}")
    
    # Validate groups
    valid_groups = {}
    for name, members in groups.items():
        valid_chs = [ch for ch in members if ch in ch_cols]
        if valid_chs:
            valid_groups[name] = valid_chs
        else:
            print(f"[group] Warning: ROI '{name}' has no valid channels, skipping")
    
    if not valid_groups:
        raise ValueError("No valid ROI groups found")
    
    group_names = list(valid_groups.keys())
    conditions = sorted(df['condition'].unique().to_list())
    base = os.path.splitext(os.path.basename(ip))[0]
    out_folder = os.path.join(os.getcwd(), f"{base}_{suffix}")
    os.makedirs(out_folder, exist_ok=True)
    
    print(f"[group] ROIs: {group_names}, Conditions: {conditions}")
    
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        epochs = cond_df['epoch_id'].unique().to_list()
        
        roi_means, roi_sems = [], []
        for roi_name in group_names:
            roi_chs = valid_groups[roi_name]
            
            # Compute mean per epoch, then stats across epochs
            epoch_means = []
            for eid in epochs:
                epoch_df = cond_df.filter(pl.col('epoch_id') == eid)
                roi_data = epoch_df.select(roi_chs).to_numpy()
                epoch_means.append(float(np.mean(roi_data)))
            
            roi_means.append(float(np.mean(epoch_means)))
            roi_sems.append(float(np.std(epoch_means, ddof=1) / np.sqrt(len(epoch_means))) if len(epoch_means) > 1 else 0.0)
        
        pl.DataFrame({
            'condition': [cond],
            'x_data': [group_names],
            'y_data': [roi_means],
            'y_var': [roi_sems],
            'plot_type': ['bar'],
            'x_label': [x_label],
            'y_label': [y_label],
            'y_ticks': [y_lim] if y_lim is not None else [None]
        }).write_parquet(os.path.join(out_folder, f"{base}_{suffix}{idx+1}.parquet"))
        print(f"[group]   {cond}: {len(epochs)} epochs, {len(group_names)} ROIs")
    
    signal_path = os.path.join(os.getcwd(), f"{base}_{suffix}.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[group] Output: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_groups(a[1], a[2],
                               float(a[3]) if len(a) > 3 and a[3] and a[3] != 'None' else None,
                               a[4] if len(a) > 4 else 'ROI',
                               a[5] if len(a) > 5 else 'Mean',
                               a[6] if len(a) > 6 else 'roi') if len(a) >= 3 else (
        print('Aggregate channels by ROIs per condition. Plot-ready output.'),
        print('[group] Usage: python group_analyzer.py <epoched.parquet> <groups_json> [y_lim] [x_label] [y_label] [suffix]'),
        print('[group] Example: python group_analyzer.py data.parquet \'{"DLPFC": ["S1_D1"], "VMPFC": ["S2_D2"]}\' 0.5 "ROI" "HbO2 (μM)" fnirs'),
        sys.exit(1)))(sys.argv)
