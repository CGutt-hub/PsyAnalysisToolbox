import sys, os, mne, warnings
from mne_nirs.channels import get_long_channels
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def channel_quality(ip: str, out: str | None = None) -> str:
    print(f"[PROC] Channel quality assessment: {ip}")
    if not ip.endswith('.fif'): print(f"[PROC] Error: Quality assessment requires MNE .fif format"); sys.exit(1)
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    picks = get_long_channels(raw)
    raw.pick(picks)
    base = os.path.splitext(os.path.basename(ip))[0]
    out_file = out or f"{base}_quality.fif"
    raw.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (quality filtered): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: channel_quality(a[1], a[2] if len(a) > 2 else None) if len(a) >= 2 else (print("Usage: python channel_quality_processor.py <input.fif> [output.fif]"), sys.exit(1)))(sys.argv)
