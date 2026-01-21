"""Channel Selector Processor - Select channels by pattern, indices, or quality criteria."""
import sys, os, mne, warnings, re
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def select_channels(ip: str, selector: str = '.*', mode: str = 'regex') -> str:
    """Select channels from .fif file.
    
    Modes:
        regex: Select channels matching regex pattern (default: .* = all)
        indices: Comma-separated indices like "0,1,5,6"
        names: Comma-separated exact names like "Fp1,Fp2,Fz"
    """
    if not os.path.exists(ip): print(f"[channel_selector] File not found: {ip}"); sys.exit(1)
    if not ip.endswith('.fif'): print(f"[channel_selector] Error: Requires .fif format"); sys.exit(1)
    print(f"[channel_selector] Channel selection: {ip}, mode={mode}, selector={selector}")
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    all_ch = raw.ch_names
    if mode == 'regex':
        picks = [i for i, ch in enumerate(all_ch) if re.match(selector, ch)]
    elif mode == 'indices':
        picks = [int(i) for i in selector.split(',')]
    elif mode == 'names':
        names = [n.strip() for n in selector.split(',')]
        picks = [i for i, ch in enumerate(all_ch) if ch in names]
    else:
        print(f"[channel_selector] Unknown mode: {mode}"); sys.exit(1)
    if not picks: print(f"[channel_selector] No channels matched selector"); sys.exit(1)
    raw.pick(picks)
    print(f"[channel_selector] Selected {len(picks)}/{len(all_ch)} channels")
    base = os.path.splitext(os.path.basename(ip))[0]
    out_file = f"{base}_sel.fif"
    raw.save(out_file, overwrite=True, verbose=False)
    print(f"[channel_selector] Output: {out_file}")
    return out_file

if __name__ == '__main__': (lambda a: select_channels(a[1], a[2] if len(a) > 2 else '.*', a[3] if len(a) > 3 else 'regex') if len(a) >= 2 else (print('[channel_selector] Select channels by regex pattern, indices, or names.\nUsage: channel_selector_processor.py <input.fif> [selector=.*] [mode:regex|indices|names]'), sys.exit(1)))(sys.argv)
