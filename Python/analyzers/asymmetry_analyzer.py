import polars as pl, numpy as np, sys, ast, os

def compute_asymmetry(ip: str, band: str, pairs: list, y_lim: float | None = None, y_label: str | None = None, suffix: str = 'asym') -> str:
    """Compute asymmetry (log ratio) between paired channels for a single condition.
    
    Args:
        ip: Input PSD parquet file (single condition, raw per-channel data)
        band: Frequency band name to use
        pairs: List of channel pairs, e.g. [('F3', 'F4'), ('F7', 'F8')]
        y_lim: Optional Y-axis maximum limit
        y_label: Optional Y-axis label (default: 'Asymmetry ({band})')
        suffix: Output file suffix (default 'asym', use 'fai' for FAI compatibility)
    """
    print(f"[asymmetry] Asymmetry analysis: {ip}, band={band}, pairs={pairs}")
    
    df = pl.read_parquet(ip)
    
    if 'x_data' in df.columns:
        print(f"[asymmetry] Error: Input must be raw PSD data with per-channel values, not plot-ready format")
        sys.exit(1)
    
    base = os.path.splitext(os.path.basename(ip))[0]
    cond = df['condition'][0] if 'condition' in df.columns else base
    label = y_label or f'Asymmetry ({band})'
    
    pair_names, asym_vals, asym_sems = [], [], []
    for left, right in pairs:
        left_data = df.filter((pl.col('channel') == left) & (pl.col('band') == band))
        right_data = df.filter((pl.col('channel') == right) & (pl.col('band') == band))
        
        if len(left_data) > 0 and len(right_data) > 0:
            left_mean = float(left_data['power'][0])
            right_mean = float(right_data['power'][0])
            left_std = float(left_data['power_std'][0]) if 'power_std' in left_data.columns else 0.0
            right_std = float(right_data['power_std'][0]) if 'power_std' in right_data.columns else 0.0
            
            # Asymmetry = ln(right) - ln(left)
            asym_mean = np.log(right_mean) - np.log(left_mean)
            # Error propagation
            asym_var = (right_std / right_mean) ** 2 + (left_std / left_mean) ** 2 if right_mean > 0 and left_mean > 0 else 0.0
            asym_sem = float(np.sqrt(asym_var))
            
            pair_names.append(f"{left}-{right}")
            asym_vals.append(float(asym_mean))
            asym_sems.append(asym_sem)
        else:
            print(f"[asymmetry]   Warning: Missing data for pair {left}-{right}")
    
    out_path = os.path.join(os.getcwd(), f"{base}_{suffix}.parquet")
    pl.DataFrame({
        'condition': [cond],
        'x_data': [pair_names],
        'y_data': [asym_vals],
        'y_var': [asym_sems],
        'plot_type': ['bar'],
        'x_label': ['Channel Pair'],
        'y_label': [label],
        'y_ticks': [y_lim] if y_lim is not None else [None]
    }).write_parquet(out_path)
    
    print(f"[asymmetry]   {cond}: {len(asym_vals)} pairs -> {out_path}")
    print(out_path)
    return out_path

if __name__ == '__main__':
    (lambda a: compute_asymmetry(a[1], a[2], ast.literal_eval(a[3]), 
                                  float(a[4]) if len(a) > 4 and a[4] and a[4] != 'None' else None,
                                  a[5] if len(a) > 5 and a[5] != 'None' else None,
                                  a[6] if len(a) > 6 else 'asym') if len(a) >= 4 else (
        print('[asymmetry] Compute log-ratio asymmetry between paired channels. Plot-ready output.\nUsage: asymmetry_analyzer.py <psd_folder> <band> <pairs_list> [y_lim] [y_label] [suffix]'),
        sys.exit(1)))(sys.argv)
