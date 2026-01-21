"""Baseline Correction Processor - Subtract baseline mean from signals."""
import polars as pl, numpy as np, sys, os

def baseline_correct(ip: str, baseline_sec: float = 5.0, sfreq: float | None = None) -> str:
    """Subtract mean of first N seconds from each channel. Generic for any time series."""
    if not os.path.exists(ip): print(f"[baseline_correction] File not found: {ip}"); sys.exit(1)
    print(f"[baseline_correction] Baseline correction: {ip}, baseline={baseline_sec}s")
    df = pl.read_parquet(ip)
    data_cols = [c for c in df.columns if c not in ['time', 'sfreq', 'condition', 'epoch_id']]
    if not data_cols: print("[baseline_correction] No data columns found"); sys.exit(1)
    fs = sfreq or (float(df['sfreq'][0]) if 'sfreq' in df.columns else 1.0)
    n_baseline = int(baseline_sec * fs)
    print(f"[baseline_correction] Using first {n_baseline} samples as baseline ({len(data_cols)} channels)")
    result = df.with_columns([(pl.col(c) - pl.col(c).head(n_baseline).mean()).alias(c) for c in data_cols])
    out_file = ip.replace('.parquet', '_bl.parquet')
    result.write_parquet(out_file)
    print(f"[baseline_correction] Output: {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: baseline_correct(a[1], float(a[2]) if len(a) > 2 else 5.0, float(a[3]) if len(a) > 3 else None) if len(a) >= 2 else (print('[baseline_correction] Subtract mean of first N seconds from each channel.\nUsage: baseline_correction_processor.py <input.parquet> [baseline_sec=5.0] [sfreq]'), sys.exit(1)))(sys.argv)
