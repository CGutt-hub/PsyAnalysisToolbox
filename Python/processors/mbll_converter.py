import sys, os, mne, warnings, mne_nirs
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def mbll_convert(ip: str, ppf: str = '6.0', out: str | None = None) -> str:
    print(f"[PROC] MBLL conversion: {ip}")
    if not ip.endswith('.fif'): print(f"[PROC] Error: MBLL requires MNE .fif format"); sys.exit(1)
    raw_od = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    raw_haemo = getattr(mne_nirs.preprocessing, 'beer_lambert_law')(raw_od, ppf=float(ppf))
    base = os.path.splitext(os.path.basename(ip))[0].replace('_od', '')
    out_file = out or f"{base}_hbo.fif"
    raw_haemo.save(out_file, overwrite=True, verbose=False)
    print(f"[PROC] Output (HbO/HbR): {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: mbll_convert(a[1], a[2] if len(a) > 2 else '6.0', a[3] if len(a) > 3 else None) if len(a) >= 2 else (print("Usage: python mbll_converter.py <input_od.fif> [ppf] [output.fif]"), sys.exit(1)))(sys.argv)
