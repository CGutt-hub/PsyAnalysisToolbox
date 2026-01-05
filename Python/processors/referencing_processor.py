import polars as pl, numpy as np, sys, mne, os, warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def apply_reference(ip: str, ref: str = 'average', out: str | None = None) -> str:
    print(f"[PROC] Referencing: {ip}"); df = pl.read_parquet(ip)
    data_cols = [c for c in df.columns if c not in ['time', 'sfreq']]
    if not data_cols: print(f"[PROC] Error: No data channels"); sys.exit(1)
    base = os.path.splitext(os.path.basename(ip))[0]
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else (10.0 if any('HbO' in c or 'HbR' in c or 'HbT' in c for c in data_cols) else 1000.0)
    
    if any('HbO' in c or 'HbR' in c or 'HbT' in c for c in data_cols):  # fNIRS baseline correction
        print(f"[PROC] Baseline correction: {len(data_cols)} fNIRS channels")
        result = df.with_columns([(df[c] - df[data_cols].head(int(5.0 * sfreq)).mean()[c]).alias(c) for c in data_cols])
        out_file = out or f"{base}_baseline.parquet"; result.write_parquet(out_file)
    else:  # EEG referencing
        print(f"[PROC] {ref} reference: {len(data_cols)} EEG channels")
        raw = mne.io.RawArray(np.array([df[c].to_numpy() for c in data_cols]), mne.create_info(data_cols, sfreq, ch_types='eeg'), verbose=False)
        raw.set_eeg_reference(ref, verbose=False)
        out_file = out or f"{base}_reref.fif"; raw.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output: {out_file}"); return out_file

if __name__ == '__main__': (lambda a: apply_reference(a[1], a[2] if len(a) > 2 else 'average', a[3] if len(a) > 3 else None) if len(a) >= 2 else (print("[PROC] Usage: python referencing_processor.py <input.parquet> [reference] [output.parquet]"), sys.exit(1)))(sys.argv)
