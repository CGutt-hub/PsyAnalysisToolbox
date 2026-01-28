import polars as pl, numpy as np, sys, ast, os

def compute_asymmetry(ip: str, pairs: list[tuple[str,str]], mode: str = 'log', 
                      band: str | None = None, y_lim: float | None = None, 
                      y_label: str | None = None, suffix: str = 'asym') -> str:
    """Compute asymmetry between paired channels/regions.
    
    Supports two input formats:
    1. Channel format: columns [channel, band, power, power_std, n_epochs] - per-channel data
    2. ROI format: columns [condition, x_data, y_data, y_var] - grouped region data
    
    Args:
        ip: Input parquet file (single condition)
        pairs: List of (left, right) pairs, e.g. [('F3', 'F4')] or [('Left', 'Right')]
        mode: 'log' for ln(R)-ln(L), 'diff' for R-L (default: 'log')
        band: Category/band filter (required for channel format, ignored for ROI format)
        y_lim: Optional Y-axis limit (symmetric around 0)
        y_label: Optional Y-axis label
        suffix: Output file suffix
    
    Returns: Path to output parquet with plot-ready asymmetry data
    """
    print(f"[asymmetry] Asymmetry analysis: {ip}, pairs={pairs}, mode={mode}")
    
    df = pl.read_parquet(ip)
    base = os.path.splitext(os.path.basename(ip))[0]
    
    # Detect input format
    if 'channel' in df.columns and 'power' in df.columns:
        # Channel format: per-channel data with band categories
        if band is None:
            print("[asymmetry] Error: band parameter required for channel format")
            sys.exit(1)
        cond = df['condition'][0] if 'condition' in df.columns else base
        data_dict = _extract_channel_data(df, band)
        label = y_label or f'Asymmetry ({band})'
    elif 'x_data' in df.columns and 'y_data' in df.columns:
        # ROI format: grouped region data
        row = df.to_dicts()[0]
        cond = row.get('condition', base)
        data_dict = _extract_roi_data(row)
        label = y_label or 'Asymmetry (R - L)'
    else:
        print(f"[asymmetry] Error: Unknown input format. Expected channel (channel,band,power) or ROI (x_data,y_data)")
        sys.exit(1)
    
    # Compute asymmetry for each pair
    # Note: data_dict values are (mean, sem) tuples
    pair_names, asym_vals, asym_sems = [], [], []
    for left, right in pairs:
        left_val, left_sem = _get_value(data_dict, left)
        right_val, right_sem = _get_value(data_dict, right)
        
        if left_val is not None and right_val is not None:
            if mode == 'log' and left_val > 0 and right_val > 0:
                # Log-ratio asymmetry: ln(R) - ln(L)
                asym = np.log(right_val) - np.log(left_val)
                # Error propagation for log difference: SE = sqrt((SE_R/R)^2 + (SE_L/L)^2)
                sem = np.sqrt((right_sem / right_val)**2 + (left_sem / left_val)**2) if right_val and left_val else 0.0
            else:
                # Simple difference: R - L
                asym = right_val - left_val
                # Error propagation for difference: SE = sqrt(SE_R^2 + SE_L^2)
                sem = np.sqrt(right_sem**2 + left_sem**2)
            
            pair_names.append(f"{left}-{right}")
            asym_vals.append(float(asym))
            asym_sems.append(float(sem))
            print(f"[asymmetry]   {left}-{right}: L={left_val:.4f}±{left_sem:.4f}, R={right_val:.4f}±{right_sem:.4f}, Asym={asym:.4f}±{sem:.4f}")
        else:
            print(f"[asymmetry]   Warning: Missing data for pair {left}-{right}")
    
    out_path = os.path.join(os.getcwd(), f"{base}_{suffix}.parquet")
    pl.DataFrame({
        'condition': [cond],
        'x_data': [pair_names],
        'y_data': [asym_vals],
        'y_var': [asym_sems],
        'plot_type': ['bar'],
        'x_label': ['Pair'],
        'y_label': [label],
        'y_ticks': [y_lim] if y_lim is not None else [None]
    }).write_parquet(out_path)
    
    print(f"[asymmetry]   {cond}: {len(asym_vals)} pairs -> {out_path}")
    print(out_path)
    return out_path

def _extract_channel_data(df: pl.DataFrame, band: str) -> dict[str, tuple[float, float]]:
    """Extract {channel: (value, SEM)} from channel format.
    
    Channel files contain columns: channel, band, power, power_std, n_epochs.
    SEM = std / sqrt(n_epochs) for proper error propagation."""
    data = {}
    for row in df.filter(pl.col('band') == band).to_dicts():
        ch = row['channel']
        power = float(row['power'])
        std = float(row.get('power_std', 0.0))
        n = int(row.get('n_epochs', 1))
        sem = std / np.sqrt(n) if n > 0 else std  # Convert std to SEM
        data[ch] = (power, sem)
    return data

def _extract_roi_data(row: dict) -> dict[str, tuple[float, float]]:
    """Extract {roi_name: (value, sem)} from ROI format.
    
    Note: y_var from group_analyzer contains SEM (standard error), not variance.
    We store it directly without squaring.
    """
    x_data = row.get('x_data', [])
    y_data = row.get('y_data', [])
    y_var = row.get('y_var', [0.0] * len(y_data))
    return {x: (float(y), float(v) if isinstance(v, (int, float)) else 0.0) 
            for x, y, v in zip(x_data, y_data, y_var)}

def _get_value(data: dict, key: str) -> tuple[float | None, float]:
    """Get value by exact key or partial match."""
    if key in data:
        return data[key]
    # Try partial match (e.g., 'Left' matches 'Left PFC')
    for k, v in data.items():
        if key.lower() in k.lower():
            return v
    return None, 0.0

if __name__ == '__main__':
    def main(args):
        if len(args) < 3:
            print('[asymmetry] Compute asymmetry between paired channels/regions.')
            print('Usage: asymmetry_analyzer.py <input.parquet> <pairs> [mode] [band] [y_lim] [y_label] [suffix]')
            print('  pairs: Python list, e.g. "[(\'F3\',\'F4\'),(\'F7\',\'F8\')]" or "[(\'Left\',\'Right\')]"')
            print('  mode: "log" for ln(R)-ln(L) (default), "diff" for R-L')
            print('  band: Required for channel format, ignored for ROI format')
            sys.exit(1)
        
        ip = args[1]
        pairs = ast.literal_eval(args[2])
        mode = args[3] if len(args) > 3 and args[3] not in ('None', '') else 'log'
        band = args[4] if len(args) > 4 and args[4] not in ('None', '') else None
        y_lim = float(args[5]) if len(args) > 5 and args[5] not in ('None', '') else None
        y_label = args[6] if len(args) > 6 and args[6] not in ('None', '') else None
        suffix = args[7] if len(args) > 7 else 'asym'
        
        compute_asymmetry(ip, pairs, mode, band, y_lim, y_label, suffix)
    
    main(sys.argv)
