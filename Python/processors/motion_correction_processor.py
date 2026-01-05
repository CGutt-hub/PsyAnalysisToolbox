import sys, os, mne, warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def motion_correct(ip: str, out: str | None = None) -> str:
    print(f"[PROC] Motion correction: {ip}")
    if not ip.endswith('.fif'): print(f"[PROC] Error: Motion correction requires MNE .fif format"); sys.exit(1)
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    base = os.path.splitext(os.path.basename(ip))[0]
    out_file = out or f"{base}_moco.fif"
    raw.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (motion corrected): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: motion_correct(a[1], a[2] if len(a) > 2 else None) if len(a) >= 2 else (print("Usage: python motion_correction_processor.py <input.fif> [output.fif]"), sys.exit(1)))(sys.argv)
