"""Rejection Processor - Reject samples based on amplitude, gradient, or flatline criteria."""
import polars as pl, numpy as np, sys, os, ast

def reject_samples(ip: str, columns: list | None = None, criterion: str = 'amplitude', threshold: float = 100e-6) -> str:
    if not os.path.exists(ip): print(f"[rejection] File not found: {ip}"); sys.exit(1)
    print(f"[rejection] Rejection: {ip}, criterion={criterion}, threshold={threshold}")
    df = pl.read_parquet(ip)
    columns = columns or [c for c in df.columns if c not in ['time', 'sfreq']]
    mask = np.ones(len(df), dtype=bool)
    for col in columns:
        sig = df[col].to_numpy()
        if criterion == 'amplitude': mask &= np.abs(sig) < threshold
        elif criterion == 'gradient': mask &= np.abs(np.gradient(sig)) < threshold
        elif criterion == 'flatline': mask &= np.std(sig) > threshold
        else: print(f"[rejection] Unknown criterion: {criterion}"); sys.exit(1)
    print(f"[rejection] Retaining {np.sum(mask)} of {len(df)} samples")
    out_file = ip.replace('.parquet', '_rej.parquet')
    df.filter(mask).write_parquet(out_file)
    print(f"[rejection] Output: {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: reject_samples(a[1], ast.literal_eval(a[2]) if len(a) > 2 and a[2] not in ('', 'None') else None, a[3] if len(a) > 3 else 'amplitude', float(a[4]) if len(a) > 4 else 100e-6) if len(a) >= 2 else (print('[rejection] Reject samples by amplitude, gradient, or flatline threshold.\nUsage: rejection_processor.py <input.parquet> [columns] [criterion=amplitude] [threshold=100e-6]'), sys.exit(1)))(sys.argv)
