import polars as pl, numpy as np, sys, mne, re
from mne_nirs.signal_enhancement import short_channel_regression
from typing import cast

def apply_short_channel_regression(ip: str, out: str | None = None) -> str:
    print(f"[PROC] Short channel regression: {ip}"); df = pl.read_parquet(ip)
    fnirs_cols = [c for c in df.columns if c not in ['time', 'sfreq']]
    if not fnirs_cols: print(f"[PROC] Error: No fNIRS channels found"); sys.exit(1)
    
    # Check if short channels exist
    short_channels = [c for c in fnirs_cols if re.search(r'(^s\d+\b)|short|_sd|_short', c, re.I)]
    if not short_channels:
        print(f"[PROC] No short channels detected, skipping regression")
        out_file = out or f"{ip.replace('.parquet', '')}_scr.parquet"
        df.write_parquet(out_file); return out_file
    
    data = np.array([df[col].to_numpy() for col in fnirs_cols])
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else 10.0
    print(f"[PROC] Applying short channel regression ({len(short_channels)} short channels)")
    
    info = mne.create_info(fnirs_cols, sfreq, ch_types='fnirs_cw_amplitude')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw_corrected = short_channel_regression(raw)
    
    # Keep in MNE format - save as .fif
    out_file = out or f"{ip.replace('.parquet', '')}_scr.fif"
    raw_corrected.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (MNE Raw): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: apply_short_channel_regression(a[1], a[2] if len(a) > 2 else None) if len(a) >= 2 else (print("[PROC] Usage: python short_channel_regression_processor.py <input.parquet> [output.parquet]"), sys.exit(1)))(sys.argv)
