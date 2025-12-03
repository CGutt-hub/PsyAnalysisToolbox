import polars as pl, numpy as np, sys, os

def analyze_hrv(ip, sfreq):
    df = pl.read_parquet(ip)
    rpeaks = df['R_Peak_Sample'].to_numpy() if 'R_Peak_Sample' in df.columns else df['rpeaks'].to_numpy()
    rr = np.diff(rpeaks) / sfreq
    sdnn, rmssd = np.std(rr), np.sqrt(np.mean(np.diff(rr) ** 2))
    out = {'x_data': ['SDNN', 'RMSSD'], 'y_data': [sdnn, rmssd], 'plot_type': 'bar', 'x_axis': 'HRV Metric', 'y_axis': 'Value (ms)'}
    out_path = f"{os.path.splitext(os.path.basename(ip))[0]}_hrv.parquet"
    pl.DataFrame([out]).write_parquet(out_path)
    return out_path

if __name__ == '__main__':
    (lambda a: analyze_hrv(a[1], float(a[2])))(sys.argv)
