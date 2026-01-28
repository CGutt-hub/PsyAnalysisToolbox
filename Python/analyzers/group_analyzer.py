import polars as pl, numpy as np, sys, os, json, fnmatch, re
from collections import defaultdict

def _match_channels(patterns: list[str], available: list[str]) -> list[str]:
    """
    Match channel patterns against available channel names.
    Supports: exact match, prefix match, glob patterns (*, ?), and regex (prefix with 're:').
    """
    matched = []
    for pattern in patterns:
        if pattern in available:
            matched.append(pattern)
        elif pattern.startswith('re:'):
            regex = re.compile(pattern[3:])
            matched.extend([ch for ch in available if regex.search(ch) and ch not in matched])
        elif '*' in pattern or '?' in pattern or '[' in pattern:
            matched.extend([ch for ch in available if fnmatch.fnmatch(ch, pattern) and ch not in matched])
        else:
            prefix_matches = [ch for ch in available if ch.startswith(pattern) and ch not in matched]
            if prefix_matches:
                matched.extend(prefix_matches)
    return matched

def _auto_detect_groups(ch_cols: list[str]) -> dict[str, list[str]]:
    """
    Auto-detect channel groups from channel naming patterns.
    Falls back to single 'All' group if no pattern detected.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    
    for ch in ch_cols:
        # Try pattern: "source-detector" (e.g., "1-1:0", "2-3")
        match = re.match(r'^(\d+)-(\d+)', ch)
        if match:
            source = match.group(1)
            groups[f'S{source}'].append(ch)
            continue
        
        # Try pattern: "S1_D1" style
        match = re.match(r'^S(\d+)_D(\d+)', ch)
        if match:
            source = match.group(1)
            groups[f'S{source}'].append(ch)
            continue
    
    return dict(groups) if groups else {'All': ch_cols}

def analyze_groups(ip: str, groups_config: str, y_lim: float | None = None, 
                   x_label: str = 'Group', y_label: str = 'Mean', suffix: str = 'grp',
                   baseline_sec: float = 2.0, subplots: str = '') -> str:
    """
    Aggregate channels by groups and compute group-level statistics per condition.
    Generic group analyzer - works on any epoched multichannel data.
    
    Args:
        ip: Input parquet with epoched data (condition, epoch_id, time, channel_cols...)
        groups_config: JSON string defining groups, or 'auto' for auto-detection.
                       Channel patterns support: exact, glob (*, ?, []), regex (re:pattern)
                       Example: {"Left": ["1-*", "2-*"], "Right": ["3-*", "4-*"]}
        y_lim: Optional Y-axis limit (symmetric around zero)
        x_label: Label for x-axis (e.g., 'ROI', 'Region')
        y_label: Label for y-axis (e.g., 'Mean Value', 'Change')
        suffix: Output file suffix (default 'grp')
        baseline_sec: Seconds at epoch start for baseline correction (default 2.0)
        subplots: Dict of subplot_name: channel_filter_regex, or empty for no subplots.
                  Example: "{'HbO': ':0$', 'HbR': ':1$'}" creates HbO and HbR subplots
    
    Returns:
        Path to signal file
    """
    print(f"[group] Group analysis: {ip}")
    
    # Parse groups config
    if groups_config.lower() == 'auto':
        groups = {}
    elif os.path.isfile(groups_config):
        with open(groups_config) as f:
            groups = json.load(f)
    else:
        groups = json.loads(groups_config)
    
    # Parse subplots config
    subplot_filters = {}
    if subplots and subplots.strip():
        try:
            subplot_filters = json.loads(subplots.replace("'", '"'))
        except:
            subplot_filters = eval(subplots)  # Handle Python dict syntax
    if not subplot_filters:
        subplot_filters = {'': ''}  # Single subplot, no filter
    
    print(f"[group] Subplots: {list(subplot_filters.keys()) if any(subplot_filters.keys()) else 'none'}")
    
    df = pl.read_parquet(ip)
    meta_cols = ['time', 'sfreq', 'epoch_id', 'condition']
    all_ch_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"[group] Total channels: {len(all_ch_cols)}")
    
    # For each subplot, filter channels and validate groups
    subplot_names = list(subplot_filters.keys())
    subplot_groups = {}  # subplot_name -> {group_name: [channels]}
    
    for sp_name, sp_filter in subplot_filters.items():
        if sp_filter:
            filter_re = re.compile(sp_filter, re.IGNORECASE)
            ch_cols = [ch for ch in all_ch_cols if filter_re.search(ch)]
            print(f"[group] Subplot '{sp_name}' filter '{sp_filter}': {len(ch_cols)} channels")
        else:
            ch_cols = all_ch_cols
        
        # Validate groups for this subplot
        valid_groups = {}
        for name, members in groups.items():
            valid_chs = _match_channels(members, ch_cols)
            if valid_chs:
                valid_groups[name] = valid_chs
        
        if not valid_groups:
            valid_groups = _auto_detect_groups(ch_cols)
        
        subplot_groups[sp_name] = valid_groups
    
    # Use first subplot's groups as reference for x-axis
    group_names = list(list(subplot_groups.values())[0].keys())
    conditions = sorted(df['condition'].unique().to_list())
    base = os.path.splitext(os.path.basename(ip))[0]
    out_folder = os.path.join(os.getcwd(), f"{base}_{suffix}")
    os.makedirs(out_folder, exist_ok=True)
    
    # Determine sampling frequency for baseline
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else None
    if sfreq is None and 'time' in df.columns:
        times = df['time'].unique().sort().to_list()
        if len(times) > 1:
            sfreq = 1.0 / (times[1] - times[0])
    baseline_samples = int(baseline_sec * sfreq) if sfreq else 0
    
    print(f"[group] Groups: {group_names}, Conditions: {conditions}")
    if baseline_samples > 0:
        print(f"[group] Baseline: {baseline_sec}s ({baseline_samples} samples)")
    
    for idx, cond in enumerate(conditions):
        cond_df = df.filter(pl.col('condition') == cond)
        epochs = cond_df['epoch_id'].unique().to_list()
        
        # Compute for each subplot
        all_y_data = []
        all_y_var = []
        
        for sp_name in subplot_names:
            valid_groups = subplot_groups[sp_name]
            roi_means, roi_sems = [], []
            
            for roi_name in group_names:
                roi_chs = valid_groups.get(roi_name, [])
                if not roi_chs:
                    roi_means.append(0.0)
                    roi_sems.append(0.0)
                    continue
                
                epoch_means = []
                for eid in epochs:
                    epoch_df = cond_df.filter(pl.col('epoch_id') == eid)
                    roi_data = epoch_df.select(roi_chs).to_numpy()
                    
                    if baseline_samples > 0 and roi_data.shape[0] > baseline_samples:
                        baseline_mean = roi_data[:baseline_samples, :].mean(axis=0, keepdims=True)
                        roi_data = roi_data - baseline_mean
                        post_baseline = roi_data[baseline_samples:, :]
                        epoch_means.append(float(np.mean(post_baseline)))
                    else:
                        epoch_means.append(float(np.mean(roi_data)))
                
                roi_means.append(float(np.mean(epoch_means)))
                roi_sems.append(float(np.std(epoch_means, ddof=1) / np.sqrt(len(epoch_means))) if len(epoch_means) > 1 else 0.0)
            
            all_y_data.append(roi_means)
            all_y_var.append(roi_sems)
        
        # Output format depends on number of subplots
        subplot_labels = [s for s in subplot_names if s]  # Filter empty names
        if len(subplot_labels) > 1:
            # Multiple subplots: nested y_data
            pl.DataFrame({
                'condition': [cond],
                'x_data': [group_names],
                'y_data': [all_y_data],
                'y_var': [all_y_var],
                'labels': [subplot_labels],
                'plot_type': ['grid'],
                'x_label': [x_label],
                'y_label': [y_label],
                'y_ticks': [y_lim] if y_lim is not None else [None]
            }).write_parquet(os.path.join(out_folder, f"{base}_{suffix}{idx+1}.parquet"))
        else:
            # Single or no subplot: flat y_data
            pl.DataFrame({
                'condition': [cond],
                'x_data': [group_names],
                'y_data': [all_y_data[0]],
                'y_var': [all_y_var[0]],
                'plot_type': ['grid'],
                'x_label': [x_label],
                'y_label': [y_label],
                'y_ticks': [y_lim] if y_lim is not None else [None]
            }).write_parquet(os.path.join(out_folder, f"{base}_{suffix}{idx+1}.parquet"))
        
        print(f"[group]   {cond}: {len(epochs)} epochs, {len(group_names)} groups × {len(subplot_labels) or 1} subplots")
    
    signal_path = os.path.join(os.getcwd(), f"{base}_{suffix}.parquet")
    pl.DataFrame({
        'signal': [1],
        'source': [os.path.basename(ip)],
        'conditions': [len(conditions)],
        'groups': [group_names],
        'subplots': [subplot_labels if subplot_labels else []],
        'folder_path': [os.path.abspath(out_folder)]
    }).write_parquet(signal_path)
    
    print(f"[group] Output: {signal_path}")
    return signal_path

if __name__ == '__main__':
    (lambda a: analyze_groups(a[1], a[2],
                               float(a[3]) if len(a) > 3 and a[3] and a[3] != 'None' else None,
                               a[4] if len(a) > 4 else 'Group',
                               a[5] if len(a) > 5 else 'Mean',
                               a[6] if len(a) > 6 else 'grp',
                               float(a[7]) if len(a) > 7 and a[7] and a[7] != 'None' else 2.0,
                               a[8] if len(a) > 8 else '') if len(a) >= 3 else (
        print('Aggregate channels by groups per condition. Plot-ready output with baseline correction.'),
        print('[group] Usage: python group_analyzer.py <epoched.parquet> <groups_json> [y_lim] [x_label] [y_label] [suffix] [baseline_sec] [subplots]'),
        print('[group] Channel patterns: exact match, glob (*, ?, []), or regex (re:pattern)'),
        print('[group] subplots: dict of name:filter for subplots (e.g., "{\'HbO\': \':0$\', \'HbR\': \':1$\'}")'),
        print('[group] Example: python group_analyzer.py data.parquet \'{"Left": ["1-*"], "Right": ["3-*"]}\' 0.5 "ROI" "Value" roi 2.0 "{\'HbO\':\':0$\'}"'),
        sys.exit(1)))(sys.argv)
