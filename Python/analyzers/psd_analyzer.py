import polars as pl, numpy as np, mne, sys, ast, os, warnings

warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def compute_psd(ip: str, bands: dict, channels: list | None = None) -> str:
    """Compute PSD from epoched data. Outputs folder structure with one file per condition."""
    print(f"[PSD] Loading: {ip}")
    df = pl.read_parquet(ip)
    
    ch_names = [c for c in df.columns if c not in ['condition', 'epoch_id', 'time']]
    sfreq = 1.0 / float(df.filter(pl.col('epoch_id') == df['epoch_id'][0]).select('time').to_numpy()[1, 0])
    epoch_ids = df['epoch_id'].unique().to_list()
    
    print(f"[PSD] Data: {len(epoch_ids)} epochs, {len(ch_names)} ch, {sfreq:.1f} Hz, Bands: {bands}")
    
    # Convert to MNE and compute PSD
    data = np.stack([df.filter(pl.col('epoch_id') == eid).select(ch_names).to_numpy().T for eid in epoch_ids])
    conditions = [df.filter(pl.col('epoch_id') == eid)['condition'][0] for eid in epoch_ids]
    cond_map = {c: i+1 for i, c in enumerate(sorted(set(conditions)))}
    events = np.column_stack([np.arange(len(epoch_ids)), np.zeros(len(epoch_ids), dtype=int), [cond_map[c] for c in conditions]]).astype(int)
    
    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=False)
    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, verbose=False)
    
    spectrum = epochs.compute_psd(method='welch', fmin=min(b[0] for b in bands.values()), 
                                  fmax=max(b[1] for b in bands.values()), 
                                  picks=channels if channels else 'all', verbose=False)
    psd_data, freqs = spectrum.get_data(return_freqs=True)
    
    # Calculate band powers
    results = [{'condition': conditions[ep], 'epoch_id': epoch_ids[ep], 'channel': ch, 'band': bn, 'power': float(np.mean(psd_data[ep, ci, (freqs >= b[0]) & (freqs <= b[1])]))}
               for ep in range(len(epoch_ids)) for ci, ch in enumerate(ch_names if not channels else channels) for bn, b in bands.items()]
    
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace = os.getcwd()
    out_folder = os.path.join(workspace, f"{base}_psd")
    os.makedirs(out_folder, exist_ok=True)
    
    result_df = pl.DataFrame(results)
    conds = sorted(result_df['condition'].unique().to_list())
    band_names = sorted(bands.keys())
    
    print(f"[PSD] Processing {len(conds)} conditions: {conds}")
    
    # Output per condition: both raw channel data and plotter format
    for idx, cond in enumerate(conds):
        cond_data = result_df.filter(pl.col('condition') == cond)
        
        # Aggregate per channel and band (average across epochs)
        raw_df = cond_data.group_by(['channel', 'band']).agg([
            pl.col('power').mean().alias('power'),
            pl.col('power').std().alias('power_std'),
            pl.col('power').count().alias('n_epochs')
        ]).with_columns(pl.lit(cond).alias('condition'))
        
        raw_df.write_parquet(os.path.join(out_folder, f"{base}_psd{idx+1}.parquet"))
        
        # Also create plotter format
        band_powers = [float(np.mean(cond_data.filter(pl.col('band') == b)['power'].to_numpy())) for b in band_names]
        band_sems = [float(np.std(cond_data.filter(pl.col('band') == b)['power'].to_numpy(), ddof=1) / np.sqrt(len(cond_data.filter(pl.col('band') == b)))) for b in band_names]
        
        pl.DataFrame({'condition': [cond], 'x_data': [band_names], 'y_data': [band_powers], 'y_var': [band_sems], 
                     'plot_type': ['bar'], 'x_label': ['Frequency Band'], 'y_label': ['Mean Power (μV²/Hz)']}).write_parquet(
            os.path.join(out_folder, f"{base}_psd{idx+1}_plot.parquet"))
        print(f"[PSD]   {cond}: {base}_psd{idx+1}.parquet (raw + plot)")
    
    pl.DataFrame({'signal': [1], 'source': [os.path.basename(ip)], 'conditions': [len(conds)], 'folder_path': [os.path.abspath(out_folder)]}).write_parquet(
        os.path.join(workspace, f"{base}_psd.parquet"))
    
    print(f"[PSD] Finished. Signal: {base}_psd.parquet")
    return os.path.join(workspace, f"{base}_psd.parquet")

if __name__ == '__main__':
    (lambda a: compute_psd(a[1], ast.literal_eval(a[2]), ast.literal_eval(a[3]) if len(a) > 3 else None) if len(a) >= 3 else (
        print('[PSD] Usage: python psd_analyzer.py <epochs.parquet> <bands_dict> [channels_list]'), sys.exit(1)))(sys.argv)
