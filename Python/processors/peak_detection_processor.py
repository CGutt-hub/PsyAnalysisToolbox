import polars as pl, numpy as np, sys, neurokit2 as nk
from typing import cast
from numpy.typing import NDArray

def detect_peaks(ip: str, col: str = 'ecg', fs: float = 1000.0, out: str | None = None) -> str:
    print(f"[PROC] Peak detection: {ip}"); df = pl.read_parquet(ip)
    target = next((c for c in df.columns if col.lower() in c.lower()), None)
    if not target: print(f"[PROC] Error: Column '{col}' not found"); sys.exit(1)
    sig: NDArray[np.float64] = df[target].to_numpy()
    time_offset = df['time'][0] if 'time' in df.columns else 0.0
    print(f"[PROC] Detecting R-peaks in {target}: {len(sig)} samples (time offset: {time_offset:.1f}s)")
    peaks_dict = nk.ecg_findpeaks(sig, sampling_rate=int(fs))
    rpeaks: NDArray[np.int64] = cast(NDArray[np.int64], peaks_dict['ECG_R_Peaks'])
    result = pl.DataFrame({'R_Peak_Sample': rpeaks, 'time': time_offset + rpeaks / fs, 'sfreq': [fs] * len(rpeaks)})
    out_file = out or f"{ip.replace('.parquet', '')}_peaks.parquet"
    result.write_parquet(out_file); print(f"[PROC] Output: {out_file} ({len(rpeaks)} peaks)"); return out_file

if __name__ == '__main__': (lambda a: detect_peaks(a[1], a[2] if len(a) > 2 else 'ecg', float(a[3]) if len(a) > 3 else 1000.0, a[4] if len(a) > 4 else None) if len(a) >= 2 else (print("[PROC] Usage: python peak_detection_processor.py <input.parquet> [column] [fs] [output.parquet]"), sys.exit(1)))(sys.argv)
