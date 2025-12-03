import sys, os, polars as pl, numpy as np
from typing import Any, Dict, List, Tuple

def _epoch_mne_and_flatten(raw, events: Dict[str, List[Tuple[float, float]]], data_path: str) -> str:
    """Epoch MNE Raw object and flatten to condition/epoch_id format."""
    import mne
    
    print(f"[PROC] MNE Raw: {len(raw.times)} samples, {len(raw.ch_names)} channels, sfreq={raw.info['sfreq']} Hz")
    print(f"[PROC] Events: {sum(len(v) for v in events.values())} epochs ({', '.join(f'{c}({len(v)})' for c, v in sorted(events.items()))})")
    
    # Create MNE events array and event_id mapping
    event_list = []
    event_id = {}
    event_counter = 1
    
    for condition, epoch_pairs in sorted(events.items()):
        if condition not in event_id:
            event_id[condition] = event_counter
            event_counter += 1
        
        for start_time, stop_time in epoch_pairs:
            # MNE events: [sample, previous_event, event_id]
            start_sample = int(start_time * raw.info['sfreq'] / 1000.0)  # Convert ms to samples
            event_list.append([start_sample, 0, event_id[condition]])
    
    mne_events = np.array(event_list, dtype=int)
    tmin = 0.0
    
    # Calculate tmax from first epoch pair
    first_condition = sorted(events.keys())[0]
    first_start, first_stop = events[first_condition][0]
    tmax = (first_stop - first_start) / 1000.0  # Convert ms to seconds
    
    print(f"[PROC] Creating MNE Epochs: tmin={tmin}s, tmax={tmax}s")
    
    # Create MNE Epochs object
    epochs_obj = mne.Epochs(raw, mne_events, event_id=event_id, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)
    
    print(f"[PROC] MNE Epochs created: {len(epochs_obj)} epochs")
    
    # Flatten to condition/epoch_id format
    dfs = []
    for condition in sorted(event_id.keys()):
        cond_epochs = epochs_obj[condition]
        for epoch_idx in range(len(cond_epochs)):
            epoch_data = cond_epochs[epoch_idx].get_data()  # Shape: (n_channels, n_times)
            times = cond_epochs.times
            
            epoch_df = pl.DataFrame({
                'condition': [condition] * len(times),
                'epoch_id': [f"{condition}_{epoch_idx}"] * len(times),
                'time': times,
                **{ch_name: epoch_data[ch_idx, :] for ch_idx, ch_name in enumerate(raw.ch_names)}
            })
            dfs.append(epoch_df)
    
    out = f"{os.path.splitext(os.path.basename(data_path))[0]}_epochs.parquet"
    if dfs:
        flat_df = pl.concat(dfs)
        flat_df.write_parquet(out)
        print(f"[PROC] Flattened: {len(flat_df)} rows")
    else:
        print("[PROC] Warning: No data in epochs")
        pl.DataFrame().write_parquet(out)
    
    print(f"[PROC] Output: {out}")
    return out

def epoch_and_flatten(data_path: str, events_path: str) -> str:
    """Epoch data and flatten to condition/epoch_id format. Accepts both Polars parquet and MNE .fif files."""
    events: Dict[str, List[Tuple[float, float]]] = pl.read_parquet(events_path)['data'][0]
    
    # Check if input is MNE .fif file
    if data_path.endswith('.fif'):
        import mne
        print(f"[PROC] Loading MNE Raw: {data_path}")
        raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        return _epoch_mne_and_flatten(raw, events, data_path)
    
    # Otherwise treat as Polars parquet
    data_df: pl.DataFrame = pl.read_parquet(data_path)
    
    if not events:
        print("[PROC] Error: No events in template")
        sys.exit(1)
    
    print(f"[PROC] Data: {len(data_df)} samples, {len(data_df.columns)-1} channels")
    print(f"[PROC] Events: {sum(len(v) for v in events.values())} epochs ({', '.join(f'{c}({len(v)})' for c, v in sorted(events.items()))})")
    
    time_col: str = 'time' if 'time' in data_df.columns else data_df.columns[0]
    data_cols: List[str] = [c for c in data_df.columns if c != time_col]
    max_val: Any = data_df[time_col].max()
    data_max: float = float(max_val) if max_val is not None else 0.0
    event_max: float = float(max(sp for eps in events.values() for _, sp in eps))
    
    # Auto-detect unit mismatch and normalize (factor of 10+ difference)
    if data_max * 10 < event_max:
        data_scale, event_scale = 1.0, 1000.0
    elif event_max * 10 < data_max:
        data_scale, event_scale = 1000.0, 1.0
    else:
        data_scale, event_scale = 1.0, 1.0
    
    print(f"[PROC] Time ranges: data={data_max:.1f}, events={event_max:.1f}")
    print(f"[PROC] Unit normalization: data×{data_scale}, events×{event_scale}")
    
    data_cols_list = list(data_cols)
    epochs = {
        c: [data_df.filter(
            (pl.col(time_col) * data_scale >= st / event_scale) & 
            (pl.col(time_col) * data_scale <= sp / event_scale)
        ).select([time_col] + data_cols_list).to_numpy().tolist() for st, sp in eps]
        for c, eps in events.items()
    }
    
    print(f"[PROC] Epoched: {sum(len(v) for v in epochs.values())} arrays")
    
    dfs = [
        pl.DataFrame({
            'condition': [c] * len(arr),
            'epoch_id': [f"{c}_{epoch_idx}"] * len(arr),
            time_col: [row[0] for row in arr],
            **{data_cols_list[i]: [row[i+1] for row in arr] for i in range(len(data_cols_list))}
        })
        for c, epoch_arrays in epochs.items()
        for epoch_idx, arr in enumerate(epoch_arrays) if arr
    ]
    
    out = f"{os.path.splitext(os.path.basename(data_path))[0]}_epochs.parquet"
    if dfs:
        flat_df = pl.concat(dfs)
        flat_df.write_parquet(out)
        print(f"[PROC] Flattened: {len(flat_df)} rows")
    else:
        print("[PROC] Warning: No data in epochs")
        pl.DataFrame().write_parquet(out)
    
    print(f"[PROC] Output: {out}")
    return out

if __name__ == '__main__': (lambda a:
    epoch_and_flatten(a[1], a[2]) if len(a) == 3 else (print('[PROC] Usage: python epoching_processor.py <data.parquet> <events.parquet>'), sys.exit(1))
)(sys.argv)
