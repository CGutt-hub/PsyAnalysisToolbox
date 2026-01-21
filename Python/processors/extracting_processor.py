import polars as pl, sys, os, mne, numpy as np, warnings
from typing import Any, cast
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

def resolve(s: str, dc: list[str]) -> list[str]:
    s = s.strip().rstrip(',').strip()
    if not dc or dc == ['empty']:
        print(f"[extracting] Warning: Input has no data columns, skipping selector '{s}'")
        return []
    if ':' in s:
        p = s.split(':')
        return dc[int(p[0].strip() or 0) - 1 if p[0].strip() else 0 : (int(p[1].strip()) if len(p) > 1 and p[1].strip() else len(dc))]
    if s.lstrip('-').isdigit():
        idx = int(s)
        return [dc[idx - 1]] if idx > 0 else [dc[len(dc) + idx]]
    matched = ([c for c in dc if c.lower() == s.lower()] or [c for c in dc if s.lower() in c.lower()] or [c for c in dc if c.lower().startswith(s.lower())])
    if not matched:
        print(f"[extracting] Warning: Selector '{s}' matched no columns from: {dc}")
        return []
    return matched[:]

def load_input(ip: str) -> tuple[pl.DataFrame, dict[str, str] | None]:
    if ip.endswith('.parquet'):
        return pl.read_parquet(ip), None
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    data = np.array(raw.get_data(), dtype=np.float64)
    ch_types = {ch: raw.get_channel_types([ch])[0] for ch in raw.ch_names}
    return pl.DataFrame({'time': raw.times, **{ch: data[i] for i, ch in enumerate(raw.ch_names)}}), ch_types

def save_fif(od: pl.DataFrame, pp: str, fp: str, chs: list[str], t: np.ndarray | None, sf: float, ch_types: dict[str, str] | None) -> None:
    od.write_parquet(pp)
    print(f"[extracting] {os.path.basename(pp)} cols={od.columns}")
    if not chs:
        mne.io.RawArray(np.array([[0.0]]), mne.create_info(['empty'], 1.0, ch_types='misc'), verbose=False).save(fp, overwrite=True, verbose=False)
    else:
        cht_list = [ch_types.get(c, 'misc') for c in chs] if ch_types else ['misc'] * len(chs)
        info = mne.create_info(chs, sf, ch_types=cast(Any, cht_list))
        mne.io.RawArray(od.select(chs).to_numpy().T, info, verbose=False).save(fp, overwrite=True, verbose=False)
    print(f"[extracting] {os.path.basename(fp)}")

run = lambda ip, sels: (lambda result, b, wr, of: (lambda df, ch_types: (os.makedirs(of, exist_ok=True), print(f"[extracting] Processing {b} with {len(sels)} selectors"), [(lambda od, pp, fp, chs, t, sf: save_fif(od, pp, fp, chs, t, sf, ch_types))(df.select(['time'] + sc) if 'time' in df.columns else df.select(sc), os.path.join(of, f"{b}_extr{i+1}.parquet"), os.path.join(of, f"{b}_extr{i+1}.fif"), [c for c in (df.select(['time'] + sc) if 'time' in df.columns else df.select(sc)).columns if c != 'time'], (df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time'].to_numpy() if 'time' in df.columns else None, float(1.0 / np.median(np.diff((df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time'].to_numpy()))) if 'time' in df.columns and len((df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time']) > 1 else 1.0) for i, s in enumerate(sels) for sc in [resolve(s, df.columns[1:] if df.columns and df.columns[0].lower() == 'time' else df.columns)] if sc], pl.DataFrame({'signal': [1], 'source': [os.path.basename(ip)], 'streams': [len(sels)], 'folder_path': [os.path.abspath(of)]}).write_parquet(os.path.join(wr, f"{b}_extr.parquet")), print(f"[extracting] Extraction finished: {b}_extr.parquet"))[-1])(result[0], result[1]))(load_input(ip), os.path.splitext(os.path.basename(ip))[0], os.getcwd(), os.path.join(os.getcwd(), f"{os.path.splitext(os.path.basename(ip))[0]}_extr"))

if __name__ == '__main__': (lambda a: run(a[1], a[2:]) if len(a) >= 3 else (print('[extracting] Extract/select columns from data files into separate outputs.\nUsage: extracting_processor.py <input.fif|parquet> <selector1> [selector2 ...]'), sys.exit(1)))(sys.argv)
