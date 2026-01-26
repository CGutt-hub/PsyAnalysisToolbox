import polars as pl, numpy as np, sys, os, json, fnmatch, re
from collections import defaultdict

def _match_channels(patterns: list[str], available: list[str]) -> list[str]:
    """
    Match channel patterns against available channel names.
    Supports: exact match, prefix match, glob patterns (*, ?), and regex (prefix with 're:').
    If no wildcards and no exact match, tries prefix matching (e.g., '1-1:1' matches '1-1:1-0').
    """
    matched = []
    for pattern in patterns:
        if pattern in available:
            # Exact match
            matched.append(pattern)
        elif pattern.startswith('re:'):
            # Regex pattern
            regex = re.compile(pattern[3:])
            matched.extend([ch for ch in available if regex.search(ch) and ch not in matched])
        elif '*' in pattern or '?' in pattern or '[' in pattern:
            # Glob/fnmatch pattern
            matched.extend([ch for ch in available if fnmatch.fnmatch(ch, pattern) and ch not in matched])
        else:
            # Try prefix match (e.g., '1-1:1' matches '1-1:1-0', '1-1:1-1', etc.)
            prefix_matches = [ch for ch in available if ch.startswith(pattern) and ch not in matched]
            if prefix_matches:
                matched.extend(prefix_matches)
    return matched

def _auto_detect_groups(ch_cols: list[str]) -> dict[str, list[str]]:
    """
    Auto-detect channel groups from channel names.
    Supports multiple naming conventions:
    - NIRx style: '1-1:1-0', '1-2:2-0' -> groups by source number (first digit)
    - MNE-NIRS style: 'S1_D1 hbo', 'S1_D2 hbr' -> groups by source number
    - Generic numbered: 'ch1', 'ch2' -> single 'All' group
    """
    groups: dict[str, list[str]] = defaultdict(list)
    
    for ch in ch_cols:
        # Try NIRx format: "source-detector:index-wavelength" e.g., "1-1:1-0"
        nirx_match = re.match(r'^(\d+)-(\d+):', ch)
        if nirx_match:
            source = nirx_match.group(1)
            groups[f'S{source}'].append(ch)
            continue
        
        # Try MNE-NIRS format: "S1_D1 hbo" or "S1_D1"
        mne_match = re.match(r'^S(\d+)_D(\d+)', ch)
        if mne_match:
            source = mne_match.group(1)
            groups[f'S{source}'].append(ch)
            continue
        
        # Try simple source-detector: "1-1", "2-3"
        simple_match = re.match(r'^(\d+)-(\d+)$', ch)
        if simple_match:
            source = simple_match.group(1)
            groups[f'S{source}'].append(ch)
            continue
    
    # If we found groups, return them
    if groups:
        return dict(groups)
    
    # Fallback: put all channels in one group
    return {'All': ch_cols}

def analyze_groups(ip: str, groups_config: str, y_lim: float | None = None, 
                   x_label: str = 'ROI', y_label: str = 'Mean', suffix: str = 'roi',
                   baseline_sec: float = 2.0) -> str:
    """
    Aggregate channels by groups (ROIs) and compute group-level statistics per condition.
    Generic group analyzer - works on epoched multichannel data.
    
    Args:
        ip: Input parquet with epoched data (condition, epoch_id, time, channel_cols...)
        groups_config: JSON string defining groups. Channel patterns support:
                       - Exact match: "S1_D1"
                       - Glob patterns: "1-*", "S?_D1", "[12]-*"
                       - Regex (prefix with 're:'): "re:^[1-4]-.*"
                       Example: {"DLPFC_L": ["1-*", "2-*"], "VMPFC": ["re:^[5-8]-"]}
        y_lim: Optional Y-axis maximum limit
        x_label: Label for x-axis (e.g., 'ROI', 'Brain Region')
        y_label: Label for y-axis (e.g., 'HbO2 Change (μM)', 'Mean Activity')
        suffix: Output file suffix (default 'roi')
        baseline_sec: Seconds at start of each epoch to use as baseline (default 2.0)
    
    Returns:
        Path to signal file
    """
    print(f"[group] ROI analysis: {ip}")
    
    # Parse groups config - "auto" triggers auto-detection
    if groups_config.lower() == 'auto':
        groups = {}  # Will trigger auto-detection below
    elif os.path.isfile(groups_config):
        with open(groups_config) as f:
            groups = json.load(f)
    else:
        groups = json.loads(groups_config)
    
    df = pl.read_parquet(ip)
    meta_cols = ['time', 'sfreq', 'epoch_id', 'condition']
    ch_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"[group] Available channels ({len(ch_cols)}): {ch_cols[:10]}..." if len(ch_cols) > 10 else f"[group] Available channels: {ch_cols}")
    print(f"[group] Looking for ROIs: {list(groups.keys())}")
    
    # Validate groups with pattern matching
    valid_groups = {}
    for name, members in groups.items():
        valid_chs = _match_channels(members, ch_cols)
        if valid_chs:
            valid_groups[name] = valid_chs
            print(f"[group]   '{name}': matched {len(valid_chs)} channels from {len(members)} patterns")
        else:
            print(f"[group] Warning: ROI '{name}' has no valid channels (patterns: {members}), skipping")
    
    # Auto-detect groups if none matched
    if not valid_groups:
        print(f"[group] No ROI groups matched. Auto-detecting from channel names...")
        valid_groups = _auto_detect_groups(ch_cols)
        if valid_groups:
            print(f"[group] Auto-detected {len(valid_groups)} groups: {list(valid_groups.keys())}")
            for name, chs in valid_groups.items():
                print(f"[group]   '{name}': {len(chs)} channels")
        else:
            raise ValueError(f"No valid ROI groups found and auto-detection failed. Available channels: {ch_cols[:5]}{'...' if len(ch_cols) > 5 else ''}")
    
    group_names = list(valid_groups.keys())
    conditions = sorted(df['condition'].unique().to_list())
    base = os.path.splitext(os.path.basename(ip))[0]
    out_folder = os.path.join(os.getcwd(), f"{base}_{suffix}")
    os.makedirs(out_folder, exist_ok=True)
    
    # Determine sampling frequency for baseline calculation
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else None
    if sfreq is None and 'time' in df.columns:
        times = df['time'].unique().sort().to_list()
        if len(times) > 1:
            sfreq = 1.0 / (times[1] - times[0])
    baseline_samples = int(baseline_sec * sfreq) if sfreq else 0
    
    print(f"[group] ROIs: {group_names}, Conditions: {conditions}")
    if baseline_samples > 0:
        print(f"[group] Per-epoch baseline correction: first {baseline_sec}s ({baseline_samples} samples)")
    
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        epochs = cond_df['epoch_id'].unique().to_list()
        
        roi_means, roi_sems = [], []
        for roi_name in group_names:
            roi_chs = valid_groups[roi_name]
            
            # Compute mean per epoch (with per-epoch baseline correction), then stats across epochs
            epoch_means = []
            for eid in epochs:
                epoch_df = cond_df.filter(pl.col('epoch_id') == eid)
                roi_data = epoch_df.select(roi_chs).to_numpy()
                
                # Per-epoch baseline correction: subtract mean of first N samples
                if baseline_samples > 0 and roi_data.shape[0] > baseline_samples:
                    baseline_mean = roi_data[:baseline_samples, :].mean(axis=0, keepdims=True)
                    roi_data = roi_data - baseline_mean
                
                # Compute mean of post-baseline period (exclude baseline samples from mean)
                if baseline_samples > 0 and roi_data.shape[0] > baseline_samples:
                    post_baseline_data = roi_data[baseline_samples:, :]
                    epoch_means.append(float(np.mean(post_baseline_data)))
                else:
                    epoch_means.append(float(np.mean(roi_data)))
            
            roi_means.append(float(np.mean(epoch_means)))
            roi_sems.append(float(np.std(epoch_means, ddof=1) / np.sqrt(len(epoch_means))) if len(epoch_means) > 1 else 0.0)
        
        pl.DataFrame({
            'condition': [cond],
            'x_data': [group_names],
            'y_data': [roi_means],
            'y_var': [roi_sems],
            'plot_type': ['grid'],  # Use 'grid' to match plotter's bar-grid layout
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
                               a[6] if len(a) > 6 else 'roi',
                               float(a[7]) if len(a) > 7 and a[7] and a[7] != 'None' else 2.0) if len(a) >= 3 else (
        print('Aggregate channels by ROIs per condition. Plot-ready output with per-epoch baseline correction.'),
        print('[group] Usage: python group_analyzer.py <epoched.parquet> <groups_json> [y_lim] [x_label] [y_label] [suffix] [baseline_sec]'),
        print('[group] Channel patterns: exact match, glob (*, ?, []), or regex (re:pattern)'),
        print('[group] baseline_sec: seconds at start of each epoch for baseline (default 2.0)'),
        print('[group] Example: python group_analyzer.py data.parquet \'{"DLPFC_L": ["1-*", "2-*"], "VMPFC": ["re:^[5-8]-"]}\' 0.5 "ROI" "HbO2 (μM)" fnirs 2.0'),
        sys.exit(1)))(sys.argv)
