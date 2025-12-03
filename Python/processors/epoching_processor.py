import sys, os, polars as pl, numpy as np, warnings
from typing import Any, Dict, List, Tuple

# Suppress MNE naming convention warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def _epoch_mne_and_flatten(raw, events: Dict[str, List[Tuple[float, float]]], data_path: str, recording_start_time: float = 0.0) -> str:
    """Epoch MNE Raw object and flatten to condition/epoch_id format.
    
    Args:
        raw: MNE Raw object
        events: Dictionary of condition -> list of (start, stop) tuples
        data_path: Path to the data file
        recording_start_time: Recording start time in seconds (for converting absolute to relative times)
    """
    import mne
    
    print(f"[PROC] MNE Raw: {len(raw.times)} samples, {len(raw.ch_names)} channels, sfreq={raw.info['sfreq']} Hz")
    print(f"[PROC] Events: {sum(len(v) for v in events.values())} epochs ({', '.join(f'{c}({len(v)})' for c, v in sorted(events.items()))})")
    
    if recording_start_time > 0:
        print(f"[PROC] Recording start time: {recording_start_time:.1f}s ({int(recording_start_time * raw.info['sfreq'])} samples)")
    
    # Unit normalization: detect if event times are in samples, seconds, or milliseconds
    all_event_times = [t for pairs in events.values() for start, stop in pairs for t in (start, stop)]
    max_event_time = max(all_event_times) if all_event_times else 0
    max_samples = len(raw.times)
    recording_duration = raw.times[-1]  # in seconds
    
    # Detect unit: Check if events are absolute samples (need offset correction) or relative
    recording_start_sample = int(recording_start_time * raw.info['sfreq'])
    
    if max_event_time > max_samples * 0.1:  # Events are in sample range
        # Check if absolute or relative samples
        if max_event_time > max_samples:  # Absolute samples (beyond recording length)
            time_scale = 1.0
            is_samples = True
            is_absolute = True
            print(f"[PROC] Events detected as absolute samples (max={max_event_time:.0f}, recording samples={max_samples})")
            print(f"[PROC] Will subtract recording start ({recording_start_sample} samples) to get relative indices")
        else:  # Relative samples
            time_scale = 1.0
            is_samples = True
            is_absolute = False
            print(f"[PROC] Events detected as relative samples (max={max_event_time:.0f}, recording samples={max_samples})")
    elif max_event_time / recording_duration > 10:  # Much larger than duration in seconds
        time_scale = 1000.0  # Events in milliseconds
        is_samples = False
        is_absolute = False
        print(f"[PROC] Events detected as milliseconds (max={max_event_time:.1f}ms, recording={recording_duration:.1f}s)")
    else:  # Events in seconds
        time_scale = 1.0
        is_samples = False
        is_absolute = False
        print(f"[PROC] Events detected as seconds (max={max_event_time:.1f}s, recording={recording_duration:.1f}s)")
    
    # Create MNE events array and event_id mapping
    event_list = []
    event_id = {}
    event_counter = 1
    
    # Check if events need time offset correction (absolute timestamps vs relative samples)
    first_event_time = min(start for pairs in events.values() for start, stop in pairs)
    
    # Try to get recording start time from the original parquet file (before .fif conversion)
    # The .fif file path like "test_analysis/EV_002_xdf4_extr4_reref_filt.fif"
    # Should map to "test_analysis/EV_002_xdf4_extr/EV_002_xdf4_extr4.parquet"
    recording_start_time = 0
    base_name = os.path.basename(data_path).split('_reref')[0].split('_regr')[0].split('_filt')[0].replace('.fif', '')
    data_dir = os.path.dirname(data_path)
    possible_parquet = os.path.join(data_dir, base_name, f"{base_name}.parquet")
    
    print(f"[PROC] Looking for original parquet: {possible_parquet}")
    if os.path.exists(possible_parquet):
        temp_df = pl.read_parquet(possible_parquet)
        if 'time' in temp_df.columns:
            recording_start_time = float(temp_df['time'][0])
            print(f"[PROC] Found original parquet file, recording start time: {recording_start_time:.3f}s")
    else:
        print(f"[PROC] Warning: Could not find original parquet file at {possible_parquet}")
        recording_start_time = 0
    
    # If events are in samples but reference absolute time, convert using recording start
    if is_samples and first_event_time > max_samples:
        # Events are absolute sample indices from timestamp * sfreq
        # Convert back to time, subtract recording start, then to relative samples
        time_offset = recording_start_time
        print(f"[PROC] Detected absolute time-based samples. Recording start: {recording_start_time:.1f}s")
        needs_offset_correction = True
    else:
        time_offset = 0
        needs_offset_correction = False
    
    for condition, epoch_pairs in sorted(events.items()):
        if condition not in event_id:
            event_id[condition] = event_counter
            event_counter += 1
        
        for start_time, stop_time in epoch_pairs:
            # MNE events: [sample, previous_event, event_id]
            if is_samples:
                if needs_offset_correction:
                    # Convert sample index back to time, subtract offset, convert to relative sample
                    start_time_sec = start_time / raw.info['sfreq']
                    relative_time = start_time_sec - time_offset
                    start_sample = int(relative_time * raw.info['sfreq'])
                else:
                    start_sample = int(start_time)  # Already relative samples
            else:
                start_sample = int(start_time * raw.info['sfreq'] / time_scale)  # Convert to samples
            
            # Validate sample is within recording
            if 0 <= start_sample < max_samples:
                event_list.append([start_sample, 0, event_id[condition]])
            else:
                print(f"[PROC] Warning: Skipping epoch for {condition} - start_sample {start_sample} out of range [0, {max_samples})")
    
    # Sort events chronologically
    event_list.sort(key=lambda x: x[0])
    mne_events = np.array(event_list, dtype=int)
    
    if len(mne_events) == 0:
        print(f"[PROC] Error: No valid events after filtering")
        return ""
    tmin = 0.0
    
    # Calculate tmax from first epoch pair
    first_condition = sorted(events.keys())[0]
    first_start, first_stop = events[first_condition][0]
    if is_samples:
        tmax = (first_stop - first_start) / raw.info['sfreq']  # Convert samples to seconds
    else:
        tmax = (first_stop - first_start) / time_scale  # Convert time units to seconds
    
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

def epoch_and_flatten(data_path: str, events_path: str, original_data_path: str | None = None) -> str:
    """Epoch data and flatten to condition/epoch_id format. Accepts .fif or .parquet. Optional original_data_path for .fif to get recording start time."""
    events: Dict[str, List[Tuple[float, float]]] = pl.read_parquet(events_path)['data'][0]
    
    if data_path.endswith('.fif'):
        import mne
        print(f"[PROC] Loading MNE Raw: {data_path}")
        raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        
        recording_start_time = 0.0
        if original_data_path and os.path.exists(original_data_path):
            recording_start_time = float(pl.read_parquet(original_data_path)['time'][0])
            print(f"[PROC] Recording starts at {recording_start_time:.1f}s (from provided path)")
        else:
            base = os.path.splitext(os.path.basename(data_path))[0].split('_reref')[0].split('_filt')[0].split('_regr')[0]
            for ppath in [os.path.join(os.path.dirname(data_path), base + '.parquet'), 
                          os.path.join(os.path.dirname(data_path), base, base + '.parquet')]:
                if os.path.exists(ppath):
                    recording_start_time = float(pl.read_parquet(ppath)['time'][0])
                    print(f"[PROC] Recording starts at {recording_start_time:.1f}s (auto-found)")
                    break
            if recording_start_time == 0:
                print(f"[PROC] Warning: Recording start time unknown, assuming 0. Provide as 3rd arg if needed.")
        
        return _epoch_mne_and_flatten(raw, events, data_path, recording_start_time)
    
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
    epoch_and_flatten(a[1], a[2], a[3] if len(a) > 3 else None) if len(a) >= 3 else 
    (print('[PROC] Usage: python epoching_processor.py <data> <events.parquet> [original_data.parquet]'), sys.exit(1))
)(sys.argv)
