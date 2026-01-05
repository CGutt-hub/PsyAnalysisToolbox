import polars as pl, numpy as np, sys, neurokit2 as nk
from typing import cast
from numpy.typing import NDArray

def detect_peaks(ip: str, fs: float) -> str:
    print(f"[PROC] Peak detection: {ip}"); df = pl.read_parquet(ip)
    # Auto-detect ECG column (look for 'ecg' in column name)
    target = next((c for c in df.columns if 'ecg' in c.lower()), None)
    if not target: print(f"[PROC] Error: No ECG column found"); sys.exit(1)
    sig: NDArray[np.float64] = df[target].to_numpy()
    time_offset = df['time'][0] if 'time' in df.columns else 0.0
    print(f"[PROC] Detecting R-peaks in {target}: {len(sig)} samples (time offset: {time_offset:.1f}s)")
    peaks_dict = nk.ecg_findpeaks(sig, sampling_rate=int(fs))
    rpeaks: NDArray[np.int64] = cast(NDArray[np.int64], peaks_dict['ECG_R_Peaks'])
    result = pl.DataFrame({'R_Peak_Sample': rpeaks, 'time': time_offset + rpeaks / fs, 'sfreq': [fs] * len(rpeaks)})
    out_file = f"{ip.replace('.parquet', '')}_peaks.parquet"
    result.write_parquet(out_file); print(f"[PROC] Output: {out_file} ({len(rpeaks)} peaks)"); return out_file

if __name__ == '__main__': (lambda a: detect_peaks(a[1], float(a[2])) if len(a) == 3 else (print("[PROC] Usage: python peak_detection_processor.py <input.parquet> <fs>"), sys.exit(1)))(sys.argv)
