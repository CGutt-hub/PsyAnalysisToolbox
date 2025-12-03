import pyxdf, polars as pl, sys, os, numpy as np

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

make_df = lambda s: (lambda ts, data, names: pl.DataFrame({'time': ts, **{names[j]: data[:, j] for j in range(len(names))}}) if names and len(ts) > 0 else pl.DataFrame({'time': ts, **{f'column_{j}': data[:, j] for j in range(data.shape[1] if hasattr(data, 'shape') else (len(data[0]) if len(data) > 0 else 0))}}) if len(ts) > 0 else pl.DataFrame({'time': [], 'empty': []}))(s.get('time_stamps', []), np.array(s.get('time_series', [])), get_ch_names(s))

def read_xdf(ip):
    print(f"[READER] Loading: {ip}")
    streams = pyxdf.load_xdf(ip)[0]
    base = os.path.splitext(os.path.basename(ip))[0]
    workspace_root = os.getcwd()
    out_folder = os.path.join(workspace_root, f"{base}_xdf")
    os.makedirs(out_folder, exist_ok=True)
    for i, s in enumerate(streams):
        df = make_df(s)
        out_path = os.path.join(out_folder, f"{base}_xdf{i+1}.parquet")
        df.write_parquet(out_path)
        print(f"[READER] Stream {i+1}/{len(streams)}: {df.shape}")
    signal_path = os.path.join(workspace_root, f"{base}_xdf.parquet")
    pl.DataFrame({'signal': [1], 'source': [os.path.basename(ip)], 'streams': [len(streams)]}).write_parquet(signal_path)
    print(f"[READER] Output: {signal_path}")

if __name__ == '__main__': (lambda a: read_xdf(a[1]) if len(a) == 2 else (print("[READER] Usage: python xdf_reader.py <input.xdf>"), sys.exit(1)))(sys.argv)

