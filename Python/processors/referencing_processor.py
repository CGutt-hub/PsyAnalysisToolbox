import polars as pl, numpy as np, sys, mne
from typing import cast
from numpy.typing import NDArray

def apply_reference(ip: str, ref: str = 'average', out: str | None = None) -> str:
    print(f"[PROC] Referencing: {ip}"); df = pl.read_parquet(ip)
    eeg_cols = [c for c in df.columns if c not in ['time', 'sfreq']]
    if not eeg_cols: print(f"[PROC] Error: No EEG channels found"); sys.exit(1)
    
    data = np.array([df[col].to_numpy() for col in eeg_cols])
    sfreq = float(df['sfreq'][0]) if 'sfreq' in df.columns else 1000.0
    print(f"[PROC] Applying {ref} reference to {len(eeg_cols)} channels")
    
    info = mne.create_info(eeg_cols, sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference(ref, verbose=False)
    
    # Keep in MNE format - save as .fif
    out_file = out or f"{ip.replace('.parquet', '')}_reref.fif"
    raw.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (MNE Raw): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: apply_reference(a[1], a[2] if len(a) > 2 else 'average', a[3] if len(a) > 3 else None) if len(a) >= 2 else (print("[PROC] Usage: python referencing_processor.py <input.parquet> [reference] [output.parquet]"), sys.exit(1)))(sys.argv)
