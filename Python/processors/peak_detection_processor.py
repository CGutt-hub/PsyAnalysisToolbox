"""Peak Detection Processor - Detect peaks in any signal column using configurable methods."""
import polars as pl, numpy as np, sys, os
from scipy.signal import find_peaks
from numpy.typing import NDArray

def detect_peaks(ip: str, column: str, fs: float, method: str = 'scipy', height: float | None = None, distance: float | None = None) -> str:
    """Detect peaks in signal. Methods: 'scipy' (general), 'ecg' (uses neurokit2 if available)."""
    print(f"[peak_detection] Peak detection: {ip}, column={column}, method={method}")
    df = pl.read_parquet(ip)
    if column not in df.columns:
        # Auto-detect by pattern
        target = next((c for c in df.columns if column.lower() in c.lower()), None)
        if not target: print(f"[peak_detection] Column not found: {column}"); sys.exit(1)
        column = target
    sig: NDArray[np.float64] = df[column].to_numpy()
    time_offset = float(df['time'][0]) if 'time' in df.columns else 0.0
    print(f"[peak_detection] Detecting peaks in {column}: {len(sig)} samples")
    
    peaks: NDArray[np.int64]
    if method == 'ecg':
        try:
            import neurokit2 as nk
            peaks_dict = nk.ecg_findpeaks(sig, sampling_rate=int(fs))
            peaks = np.array(peaks_dict['ECG_R_Peaks'], dtype=np.int64)
        except ImportError:
            print("[peak_detection] neurokit2 not available, falling back to scipy")
            kwargs = {}
            if height is not None: kwargs['height'] = height
            if distance is not None: kwargs['distance'] = int(distance * fs)
            peaks, _ = find_peaks(sig, **kwargs)
            peaks = peaks.astype(np.int64)
    else:  # scipy
        kwargs = {}
        if height is not None: kwargs['height'] = height
        if distance is not None: kwargs['distance'] = int(distance * fs)
        peaks, _ = find_peaks(sig, **kwargs)
        peaks = peaks.astype(np.int64)
    
    result = pl.DataFrame({'peak_sample': peaks, 'time': time_offset + peaks / fs, 'sfreq': [fs] * len(peaks)})
    out_file = ip.replace('.parquet', '_peaks.parquet')
    result.write_parquet(out_file)
    print(f"[peak_detection] Output: {out_file} ({len(peaks)} peaks)")
    return out_file

if __name__ == '__main__': (lambda a: detect_peaks(a[1], a[2], float(a[3]), a[4] if len(a) > 4 else 'scipy', float(a[5]) if len(a) > 5 and a[5] else None, float(a[6]) if len(a) > 6 and a[6] else None) if len(a) >= 4 else (print('[peak_detection] Detect peaks in signal using scipy or neurokit2 (ECG R-peaks).\nUsage: peak_detection_processor.py <input.parquet> <column> <fs> [method=scipy|ecg] [height] [distance_sec]'), sys.exit(1)))(sys.argv)
