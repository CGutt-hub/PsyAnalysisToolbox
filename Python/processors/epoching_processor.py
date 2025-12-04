import sys, os, polars as pl, numpy as np, warnings
from typing import Dict, List, Tuple
from decimal import Decimal

warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def _epoch_mne(raw, events: Dict[str, List[Tuple[float, float]]], data_path: str, rec_start: float = 0.0) -> str:
    import mne
    
    print(f"[PROC] MNE Raw: {len(raw.times)} samples, {len(raw.ch_names)} ch, {raw.info['sfreq']} Hz")
    print(f"[PROC] Events: {sum(len(v) for v in events.values())} epochs")
    
    # Detect event time format
    all_times = [t for pairs in events.values() for start, stop in pairs for t in (start, stop)]
    max_event, max_samples = max(all_times), len(raw.times)
    rec_start_sample = int(rec_start * raw.info['sfreq'])
    
    # Determine if events are absolute samples (need offset), relative samples, seconds, or milliseconds
    if max_event > max_samples * 0.1:  # Sample range
        is_absolute = max_event > max_samples
        convert = lambda t: int((t / raw.info['sfreq'] - rec_start) * raw.info['sfreq']) if is_absolute else int(t)
        print(f"[PROC] Events: {'absolute' if is_absolute else 'relative'} samples (offset: {rec_start_sample if is_absolute else 0})")
    elif max_event / raw.times[-1] > 10:  # Milliseconds
        convert = lambda t: int(t * raw.info['sfreq'] / 1000.0)
        print(f"[PROC] Events: milliseconds")
    else:  # Seconds
        convert = lambda t: int(t * raw.info['sfreq'])
        print(f"[PROC] Events: seconds")
    
    # Build MNE events array
    event_id, event_counter, event_list = {}, 1, []
    for condition, pairs in sorted(events.items()):
        if condition not in event_id:
            event_id[condition] = event_counter
            event_counter += 1
        for start, stop in pairs:
            sample = convert(start)
            if 0 <= sample < max_samples:
                event_list.append([sample, 0, event_id[condition]])
            else:
                print(f"[PROC] Warning: {condition} epoch at sample {sample} out of range")
    
    if not event_list:
        print(f"[PROC] Error: No valid events")
        return ""
    
    mne_events = np.array(sorted(event_list, key=lambda x: x[0]), dtype=int)
    
    # Calculate epoch duration from first pair
    first_start, first_stop = events[sorted(events.keys())[0]][0]
    tmax = (convert(first_stop) - convert(first_start)) / raw.info['sfreq']
    
    print(f"[PROC] Epoching: 0-{tmax:.1f}s, {len(mne_events)} valid events")
    
    # Create and flatten epochs
    epochs_obj = mne.Epochs(raw, mne_events, event_id=event_id, tmin=0.0, tmax=tmax, 
                           baseline=None, preload=True, verbose=False)
    
    print(f"[PROC] Created: {len(epochs_obj)} epochs")
    
    dfs = []
    for cond in sorted(event_id.keys()):
        for idx, epoch_data in enumerate(epochs_obj[cond].get_data()):
            dfs.append(pl.DataFrame({
                'condition': [cond] * len(epochs_obj.times),
                'epoch_id': [f"{cond}_{idx}"] * len(epochs_obj.times),
                'time': epochs_obj.times,
                **{ch: epoch_data[i, :] for i, ch in enumerate(raw.ch_names)}
            }))
    
    out = f"{os.path.splitext(os.path.basename(data_path))[0]}_epochs.parquet"
    (pl.concat(dfs) if dfs else pl.DataFrame()).write_parquet(out)
    print(f"[PROC] Output: {out} ({len(pl.concat(dfs)) if dfs else 0} rows)")
    return out

def epoch_and_flatten(data_path: str, events_path: str, orig_path: str | None = None) -> str:
    events = pl.read_parquet(events_path)['data'][0]
    
    if data_path.endswith('.fif'):
        import mne
        print(f"[PROC] Loading: {data_path}")
        raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        
        # Get recording start time
        rec_start = 0.0
        if orig_path and os.path.exists(orig_path):
            rec_start = float(pl.read_parquet(orig_path)['time'][0])
        else:
            base = os.path.splitext(os.path.basename(data_path))[0].split('_reref')[0].split('_filt')[0].split('_regr')[0]
            for path in [f"{os.path.dirname(data_path)}/{base}.parquet", f"{os.path.dirname(data_path)}/{base}/{base}.parquet"]:
                if os.path.exists(path):
                    rec_start = float(pl.read_parquet(path)['time'][0])
                    break
        
        if rec_start > 0:
            print(f"[PROC] Recording start: {rec_start:.1f}s")
        return _epoch_mne(raw, events, data_path, rec_start)
    
    # Parquet data
    df = pl.read_parquet(data_path)
    time_col = 'time' if 'time' in df.columns else df.columns[0]
    data_cols = [c for c in df.columns if c != time_col]
    
    print(f"[PROC] Data: {len(df)} samples, {len(data_cols)} ch")
    print(f"[PROC] Events: {sum(len(v) for v in events.values())} epochs")
    
    # Unit normalization - cast to numeric types explicitly
    data_max_val = df[time_col].max()
    if isinstance(data_max_val, (int, float)):
        data_max = float(data_max_val)
    elif isinstance(data_max_val, Decimal):
        data_max = float(data_max_val)
    elif isinstance(data_max_val, str):
        data_max = float(data_max_val)
    else:
        data_max = 0.0
    event_max = float(max(sp for eps in events.values() for _, sp in eps))
    scale_data, scale_event = (1.0, 1000.0) if data_max * 10 < event_max else (1000.0, 1.0) if event_max * 10 < data_max else (1.0, 1.0)
    
    print(f"[PROC] Time ranges: data={data_max:.1f}, events={event_max:.1f}, scales: {scale_data}×{scale_event}")
    
    dfs = [
        pl.DataFrame({
            'condition': [c] * len(arr),
            'epoch_id': [f"{c}_{i}"] * len(arr),
            time_col: [r[0] for r in arr],
            **{data_cols[j]: [r[j+1] for r in arr] for j in range(len(data_cols))}
        })
        for c, pairs in events.items()
        for i, arr in enumerate([
            df.filter((pl.col(time_col) * scale_data >= st / scale_event) & 
                     (pl.col(time_col) * scale_data <= sp / scale_event))
            .select([time_col] + data_cols).to_numpy().tolist() 
            for st, sp in pairs
        ]) if arr
    ]
    
    out = f"{os.path.splitext(os.path.basename(data_path))[0]}_epochs.parquet"
    (pl.concat(dfs) if dfs else pl.DataFrame()).write_parquet(out)
    print(f"[PROC] Output: {out} ({len(pl.concat(dfs)) if dfs else 0} rows)")
    return out

if __name__ == '__main__': (lambda a:
    epoch_and_flatten(a[1], a[2], a[3] if len(a) > 3 else None) if len(a) >= 3 else 
    (print('[PROC] Usage: python epoching_processor.py <data> <events.parquet> [original.parquet]'), sys.exit(1))
)(sys.argv)
