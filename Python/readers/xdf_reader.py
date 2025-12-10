import pyxdf, polars as pl, sys, os, numpy as np, mne, warnings

# Suppress MNE naming convention warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def get_ch_names(stream):
    info = stream.get('info', {})
    desc = info.get('desc')
    if not desc: return None
    desc = desc[0] if isinstance(desc, list) and desc else desc
    if not isinstance(desc, dict) or 'channels' not in desc: return None
    channels = desc['channels']
    if isinstance(channels, list) and channels: channels = channels[0]
    if not isinstance(channels, dict) or 'channel' not in channels: return None
    ch_list = channels['channel']
    if not ch_list: return None
    if isinstance(ch_list, list):
        return [ch.get('label', [f'col{i}'])[0] if isinstance(ch.get('label'), list) else ch.get('label', f'col{i}') for i, ch in enumerate(ch_list)]
    return [ch_list.get('label', ['col0'])[0] if isinstance(ch_list.get('label'), list) else ch_list.get('label', 'col0')]

def get_stream_type(stream):
    info = stream.get('info', {})
    return info.get('type', [None])[0] if isinstance(info.get('type'), list) else info.get('type', None)

make_df = lambda s: (lambda ts, data, names: pl.DataFrame({'time': ts, **{names[j]: data[:, j] for j in range(len(names))}}) if names and len(ts) > 0 else pl.DataFrame({'time': ts, **{f'column_{j}': data[:, j] for j in range(data.shape[1] if hasattr(data, 'shape') else (len(data[0]) if len(data) > 0 else 0))}}) if len(ts) > 0 else pl.DataFrame({'time': [], 'empty': []}))(s.get('time_stamps', []), np.array(s.get('time_series', [])), get_ch_names(s))

def save_as_mne(stream, out_path, stream_type):
    """Save stream as MNE .fif file for EEG/NIRS/fNIRS data"""
    ts = stream.get('time_stamps', [])
    data = np.array(stream.get('time_series', []))
    ch_names = get_ch_names(stream)
    
    if len(ts) == 0 or data.shape[0] == 0 or not ch_names or len(ch_names) == 0:
        return False
    
    # Determine channel type
    if stream_type in ['NIRS', 'fNIRS']:
        ch_type = 'fnirs_cw_amplitude'  # fNIRS continuous wave amplitude
    elif stream_type == 'EEG':
        ch_type = 'eeg'
    else:
        ch_type = 'misc'
    
    # Calculate sampling frequency
    sfreq = float(stream.get('info', {}).get('nominal_srate', [1.0])[0] if isinstance(stream.get('info', {}).get('nominal_srate'), list) else stream.get('info', {}).get('nominal_srate', 1.0))
    
    # Create MNE info and Raw object
    info = mne.create_info(ch_names, sfreq, ch_types=ch_type)
    raw = mne.io.RawArray(data.T, info, verbose=False)
    
    # Save as .fif
    raw.save(out_path, overwrite=True, verbose=False)
    return True

def read_xdf(ip):
    print(f"[READER] Loading: {ip}")
    streams = pyxdf.load_xdf(ip)[0]
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_xdf")
    os.makedirs(out_folder, exist_ok=True)
    
    for i, s in enumerate(streams):
        stream_type = get_stream_type(s)
        fif_path = os.path.join(out_folder, f"{base}_xdf{i+1}.fif")
        parquet_path = os.path.join(out_folder, f"{base}_xdf{i+1}.parquet")
        # Save as MNE .fif (always attempt)
        success = save_as_mne(s, fif_path, stream_type)
        if success:
            ch_names = get_ch_names(s)
            n_samples = len(s.get('time_stamps', []))
            n_channels = len(ch_names) if ch_names else 0
            print(f"[READER] Stream {i+1}/{len(streams)} ({stream_type}): {n_samples} samples, {n_channels} channels -> .fif")
        else:
            print(f"[READER] Stream {i+1}/{len(streams)} ({stream_type}): Empty or not suitable for .fif, skipped .fif")
        # Save as parquet (always attempt)
        df = make_df(s)
        df.write_parquet(parquet_path)
        print(f"[READER] Stream {i+1}/{len(streams)} ({stream_type}): {df.shape} -> .parquet")
    
    signal_path = os.path.join(workspace_root, f"{base}_xdf.parquet")
    pl.DataFrame({'signal': [1], 'source': [os.path.basename(ip)], 'streams': [len(streams)]}).write_parquet(signal_path)
    print(f"[READER] Output: {signal_path}")

if __name__ == '__main__': (lambda a: read_xdf(a[1]) if len(a) == 2 else (print("[READER] Usage: python xdf_reader.py <input.xdf>"), sys.exit(1)))(sys.argv)

