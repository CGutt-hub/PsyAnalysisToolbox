import numpy as np, sys, os, mne, warnings
from numpy.typing import NDArray
from typing import cast
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def od_convert(ip: str, out: str | None = None) -> str:
    print(f"[PROC] OD conversion: {ip}")
    if not ip.endswith('.fif'): print(f"[PROC] Error: OD conversion requires MNE .fif format"); sys.exit(1)
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    data = cast(NDArray[np.float64], raw.get_data())
    baseline_samples = int(5.0 * raw.info['sfreq'])
    baseline_mean = data[:, :baseline_samples].mean(axis=1, keepdims=True)
    od_data = -np.log10(data / baseline_mean)
    od_raw = mne.io.RawArray(od_data, raw.info, verbose=False)
    base = os.path.splitext(os.path.basename(ip))[0]
    out_file = out or f"{base}_od.fif"
    od_raw.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (OD): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: od_convert(a[1], a[2] if len(a) > 2 else None) if len(a) >= 2 else (print("Usage: python od_converter.py input.fif [output.fif]"), sys.exit(1)))(sys.argv)
