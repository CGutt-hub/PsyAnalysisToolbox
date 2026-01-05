import polars as pl, sys, os, mne, numpy as np, warnings
warnings.filterwarnings('ignore', message='.*does not conform to MNE naming conventions.*')

resolve = lambda s, dc: [] if not s or s == '' else (lambda p: dc[int(p[0].strip() or 0) - 1 if p[0].strip() else 0 : (int(p[1].strip()) if p[1].strip() else len(dc))] if len(p) > 1 else dc[int(p[0].strip() or 0) - 1 if p[0].strip() else 0:])(s.split(':')) if ':' in s else [dc[int(s) - 1]] if s.lstrip('-').isdigit() and 1 <= int(s) <= len(dc) else [dc[len(dc) + int(s)]] if s.lstrip('-').isdigit() and int(s) < 0 and abs(int(s)) <= len(dc) else [c for c in dc if c.lower() == s.lower()] or [c for c in dc if s.lower() in c.lower()] or [c for c in dc if c.lower().startswith(s.lower())] or []

def load_input(ip: str) -> pl.DataFrame:
    if ip.endswith('.parquet'):
        return pl.read_parquet(ip)
    raw = mne.io.read_raw_fif(ip, preload=True, verbose=False)
    data = np.array(raw.get_data(), dtype=np.float64)
    return pl.DataFrame({'time': raw.times, **{ch: data[i] for i, ch in enumerate(raw.ch_names)}})

run = lambda ip, sels: (lambda df, b, wr, of: (os.makedirs(of, exist_ok=True), [((lambda od, pp, fp, chs, t, sf: (od.write_parquet(pp), print(f"[PROC] {os.path.basename(pp)} cols={od.columns}"), mne.io.RawArray(np.array([[0.0]]), mne.create_info(['empty'], 1.0, ch_types='misc'), verbose=False).save(fp, overwrite=True, verbose=False) if not chs else mne.io.RawArray(od.select(chs).to_numpy().T, mne.create_info(chs, sf, ch_types='misc'), verbose=False).save(fp, overwrite=True, verbose=False), print(f"[PROC] {os.path.basename(fp)}"))[-1])(df.select(['time'] + sc) if 'time' in df.columns else df.select(sc), os.path.join(of, f"{b}_extr{i+1}.parquet"), os.path.join(of, f"{b}_extr{i+1}.fif"), [c for c in (df.select(['time'] + sc) if 'time' in df.columns else df.select(sc)).columns if c != 'time'], (df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time'].to_numpy() if 'time' in df.columns else None, 1.0 / np.median(np.diff((df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time'].to_numpy())) if 'time' in df.columns and len((df.select(['time'] + sc) if 'time' in df.columns else df.select(sc))['time']) > 1 else 1.0)) for i, s in enumerate(sels) for sc in [resolve(s, df.columns[1:] if df.columns and df.columns[0].lower() == 'time' else df.columns)] if sc], pl.DataFrame({'signal': [1], 'source': [os.path.basename(ip)], 'streams': [len(sels)], 'folder_path': [os.path.abspath(of)]}).write_parquet(os.path.join(wr, f"{b}_extr.parquet")), print(f"[PROC] Extraction finished: {b}_extr.parquet"))[-1])(load_input(ip), os.path.splitext(os.path.basename(ip))[0], os.getcwd(), os.path.join(os.getcwd(), f"{os.path.splitext(os.path.basename(ip))[0]}_extr"))

if __name__ == '__main__': (lambda a: run(a[1], a[2:]) if len(a) >= 3 else (print('[PROC] Usage: python extracting_processor.py <input.parquet> <selector1> [selector2 ...]'), sys.exit(1)))(sys.argv)
