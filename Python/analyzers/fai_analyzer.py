import polars as pl, numpy as np, sys, ast, os

def compute_fai(ip: str, band: str, pairs: list) -> str:
    """Compute FAI from PSD folder structure. Outputs one file per condition."""
    print(f"[FAI] FAI analysis: {ip}, Band: {band}, Pairs: {pairs}")
    
    # Find PSD folder
    if ip.endswith('.parquet') and not os.path.isdir(ip):
        folder = os.path.join(os.path.dirname(ip) or '.', os.path.splitext(os.path.basename(ip))[0])
    else:
        folder = ip
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"PSD folder not found: {folder}")
    
    base = os.path.basename(folder)
    workspace = os.getcwd()
    out_folder = os.path.join(workspace, f"{base}_fai")
    os.makedirs(out_folder, exist_ok=True)
    
    psd_files = sorted([f for f in os.listdir(folder) if f.endswith('.parquet') and '_plot' not in f])
    conds = []
    
    for idx, psd_file in enumerate(psd_files):
        df = pl.read_parquet(os.path.join(folder, psd_file))
        
        # Check if this is plotter format or raw PSD data
        if 'x_data' in df.columns:
            # Plotter format - need to extract from raw PSD data stored elsewhere
            # FAI needs channel-level data, not aggregated band data
            print(f"[FAI] Error: Input must be raw PSD folder with per-channel data, not plotter format")
            sys.exit(1)
        
        cond = df['condition'][0] if 'condition' in df.columns else f"cond_{idx+1}"
        conds.append(cond)
        
        pair_names, fai_vals, fai_sems = [], [], []
        for left, right in pairs:
            # Get mean power and std across epochs for each channel
            left_data = df.filter((pl.col('channel') == left) & (pl.col('band') == band))
            right_data = df.filter((pl.col('channel') == right) & (pl.col('band') == band))
            
            if len(left_data) > 0 and len(right_data) > 0:
                # FAI per epoch would require individual epoch data, but we have aggregated power
                # Use error propagation: Var(ln(R) - ln(L)) ≈ Var(R)/R² + Var(L)/L²
                left_mean, left_std = float(left_data['power'][0]), float(left_data['power_std'][0])
                right_mean, right_std = float(right_data['power'][0]), float(right_data['power_std'][0])
                
                fai_mean = np.log(right_mean) - np.log(left_mean)
                fai_var = (right_std / right_mean) ** 2 + (left_std / left_mean) ** 2
                fai_sem = float(np.sqrt(fai_var))
                
                pair_names.append(f"{left}-{right}")
                fai_vals.append(float(fai_mean))
                fai_sems.append(fai_sem)
        
        pl.DataFrame({'condition': [cond], 'x_data': [pair_names], 'y_data': [fai_vals], 'y_var': [fai_sems],
                     'plot_type': ['bar'], 'x_label': ['Electrode Pair'], 'y_label': [f'FAI ({band})']}).write_parquet(
            os.path.join(out_folder, f"{base}_fai{idx+1}.parquet"))
        print(f"[FAI]   {cond}: {base}_fai{idx+1}.parquet ({len(fai_vals)} pairs)")
    
    pl.DataFrame({'signal': [1], 'source': [os.path.basename(folder)], 'conditions': [len(conds)], 'folder_path': [os.path.abspath(out_folder)]}).write_parquet(
        os.path.join(workspace, f"{base}_fai.parquet"))
    
    print(f"[FAI] Finished. Signal: {base}_fai.parquet")
    return os.path.join(workspace, f"{base}_fai.parquet")

if __name__ == '__main__':
    (lambda a: compute_fai(a[1], a[2], ast.literal_eval(a[3])) if len(a) >= 4 else (
        print('[FAI] Usage: python fai_analyzer.py <psd_folder_or_signal> <band> <pairs>'),
        print('[FAI] Example: python fai_analyzer.py data_psd.parquet alpha "[(\'F3\', \'F4\'), (\'F7\', \'F8\')]"'),
        sys.exit(1)))(sys.argv)
