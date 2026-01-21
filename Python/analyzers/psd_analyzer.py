import polars as pl, numpy as np, sys, ast, os
from scipy import signal

def compute_psd(ip: str, bands: dict, channels: list | None = None, y_lim: float | None = None) -> str:
    """Compute PSD from epoched data using scipy.signal.welch. No MNE dependency.
    
    Args:
        ip: Input parquet with epoched data [condition, epoch_id, time, channel_cols...]
        bands: Dictionary of frequency bands, e.g. {'alpha': [8, 12], 'beta': [13, 30]}
        channels: Optional list of channels to analyze
        y_lim: Optional Y-axis maximum limit
    """
    print(f"[psd] Loading: {ip}")
    df = pl.read_parquet(ip)
    
    ch_names = [c for c in df.columns if c not in ['condition', 'epoch_id', 'time']]
    if channels:
        ch_names = [c for c in ch_names if c in channels]
    
    # Detect sampling frequency from time column
    first_epoch = df.filter(pl.col('epoch_id') == df['epoch_id'][0])
    times = first_epoch['time'].to_numpy()
    dt = float(times[1]) - float(times[0]) if len(times) > 1 else 1.0/256.0
    sfreq = 1.0 / dt
    
    epoch_ids = df['epoch_id'].unique().to_list()
    conditions = [str(df.filter(pl.col('epoch_id') == eid)['condition'][0]) for eid in epoch_ids]
    
    print(f"[psd] Data: {len(epoch_ids)} epochs, {len(ch_names)} ch, {sfreq:.1f} Hz, Bands: {list(bands.keys())}")
    
    # Compute PSD per epoch per channel using scipy
    results = []
    nperseg = min(256, len(times))
    
    for ep_idx, eid in enumerate(epoch_ids):
        epoch_df = df.filter(pl.col('epoch_id') == eid)
        cond = conditions[ep_idx]
        
        for ch in ch_names:
            data = epoch_df[ch].to_numpy()
            freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
            
            for band_name, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                power = float(np.mean(psd[mask])) if mask.any() else 0.0
                results.append({
                    'condition': cond,
                    'epoch_id': eid,
                    'channel': ch,
                    'band': band_name,
                    'power': power
                })
    
    base = os.path.splitext(os.path.basename(ip))[0]
    out_folder = os.path.join(os.getcwd(), f"{base}_psd")
    os.makedirs(out_folder, exist_ok=True)
    
    result_df = pl.DataFrame(results)
    conds = sorted(result_df['condition'].unique().to_list())
    band_names = sorted(bands.keys())
    
    print(f"[psd] Processing {len(conds)} conditions")
    
    for idx, cond in enumerate(conds):
        cond_data = result_df.filter(pl.col('condition') == cond)
        
        # Raw data per channel/band
        raw_df = cond_data.group_by(['channel', 'band']).agg([
            pl.col('power').mean().alias('power'),
            pl.col('power').std().alias('power_std'),
            pl.col('power').count().alias('n_epochs')
        ]).with_columns(pl.lit(cond).alias('condition'))
        
        raw_df.write_parquet(os.path.join(out_folder, f"{base}_psd{idx+1}.parquet"))
        
        # Plotter format (aggregated across channels)
        band_powers = [float(np.mean(cond_data.filter(pl.col('band') == b)['power'].to_numpy())) for b in band_names]
        n_vals = [len(cond_data.filter(pl.col('band') == b)) for b in band_names]
        band_sems = [float(np.std(cond_data.filter(pl.col('band') == b)['power'].to_numpy(), ddof=1) / np.sqrt(n)) if n > 1 else 0.0 for b, n in zip(band_names, n_vals)]
        
        pl.DataFrame({
            'condition': [cond],
            'x_data': [band_names],
            'y_data': [band_powers],
            'y_var': [band_sems],
            'plot_type': ['bar'],
            'x_label': ['Frequency Band'],
            'y_label': ['Power (μV²/Hz)'],
            'y_ticks': [y_lim] if y_lim is not None else [None]
        }).write_parquet(os.path.join(out_folder, f"{base}_psd{idx+1}_plot.parquet"))
        print(f"[psd]   {cond}: {base}_psd{idx+1}.parquet")
    
    signal_path = os.path.join(os.getcwd(), f"{base}_psd.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conds)],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[psd] Output: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: compute_psd(a[1], ast.literal_eval(a[2]), 
                          ast.literal_eval(a[3]) if len(a) > 3 and a[3] and a[3] not in ['None', 'null'] else None,
                          float(a[4]) if len(a) > 4 and a[4] else None) if len(a) >= 3 else (
        print('Power spectral density via Welch method per frequency band. Plot-ready output.\n[psd] Usage: python psd_analyzer.py <epochs.parquet> <bands_dict> [channels] [y_lim]'), sys.exit(1)))(sys.argv)
