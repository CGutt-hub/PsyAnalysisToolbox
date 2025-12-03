import polars as pl, numpy as np, sys, scipy.signal, os
from typing import cast
from numpy.typing import NDArray

def bandpass(sig: NDArray[np.float64], lf: float, hf: float, fs: float, order: int = 2) -> NDArray[np.float64]:
    if not (0 < lf < hf < fs/2): return sig
    sos = scipy.signal.butter(order, [lf, hf], btype='band', fs=fs, output='sos')
    return cast(NDArray[np.float64], scipy.signal.sosfiltfilt(sos, sig))

def lowpass(sig: NDArray[np.float64], hf: float, fs: float, order: int = 2) -> NDArray[np.float64]:
    if not (0 < hf < fs/2): return sig
    sos = scipy.signal.butter(order, hf, btype='low', fs=fs, output='sos')
    return cast(NDArray[np.float64], scipy.signal.sosfiltfilt(sos, sig))

def highpass(sig: NDArray[np.float64], lf: float, fs: float, order: int = 2) -> NDArray[np.float64]:
    if not (0 < lf < fs/2): return sig
    sos = scipy.signal.butter(order, lf, btype='high', fs=fs, output='sos')
    return cast(NDArray[np.float64], scipy.signal.sosfiltfilt(sos, sig))

def filter_signal(ip: str, col: str, lf: str, hf: str, fs: float = 1000.0, ftype: str = 'bandpass', out: str | None = None) -> str:
    print(f"[PROC] Filtering: {ip}"); df = pl.read_parquet(ip)
    target = next((c for c in df.columns if col.lower() in c.lower()), None)
    if not target: print(f"[PROC] Error: Column '{col}' not found"); sys.exit(1)
    sig: NDArray[np.float64] = df[target].to_numpy()
    print(f"[PROC] {ftype} filter on {target}: {len(sig)} samples")
    filtered = bandpass(sig, float(lf), float(hf), float(fs)) if ftype == 'bandpass' else lowpass(sig, float(hf), float(fs)) if ftype == 'lowpass' else highpass(sig, float(lf), float(fs))
    result = pl.DataFrame({'time': df['time'] if 'time' in df.columns else np.arange(len(filtered))/float(fs), target.lower(): filtered, 'sfreq': [float(fs)]*len(filtered)})
    base = os.path.splitext(os.path.basename(ip))[0]
    out_file = f"{base}_filt.parquet"
    result.write_parquet(out_file); print(f"[PROC] Output: {out_file}"); return out_file

if __name__ == '__main__': (lambda a: filter_signal(a[1], a[2], a[3], a[4], float(a[5]) if len(a) > 5 else 1000.0, a[6] if len(a) > 6 else 'bandpass') if len(a) >= 5 else (print("[PROC] Usage: python filtering_processor.py <input.parquet> <column> <l_freq> <h_freq> [fs] [ftype]"), sys.exit(1)))(sys.argv)