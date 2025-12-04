import polars as pl, numpy as np, sys, os, ast
from scipy.signal import hilbert, butter, filtfilt
from numpy.typing import NDArray
from typing import Any

def compute_plv(stream_paths: list[str], stream_configs: list[dict[str, Any]]) -> str:
    """
    Compute PLV between arbitrary number of streams.
    
    Args:
        stream_paths: List of paths to epoched stream files
        stream_configs: List of dicts with 'type', 'channels'/'column', 'freq_band', 'sfreq' for each stream
                       type: 'continuous' (EEG, EDA) or 'event' (HRV R-peaks)
    """
    print(f"[PLV] Loading {len(stream_paths)} streams")
    for i, (path, cfg) in enumerate(zip(stream_paths, stream_configs)):
        print(f"[PLV]   Stream {i+1}: {os.path.basename(path)} ({cfg['type']})")
    
    eeg_df = pl.read_parquet(eeg_epochs)
    hrv_df = pl.read_parquet(hrv_epochs)
    
    # Filter to band of interest for EEG
    from scipy.signal import butter, filtfilt
    butter_result = butter(4, freq_band, btype='band', fs=1000.0)
    b: NDArray[np.float64] = butter_result[0]  # type: ignore[assignment]
    a: NDArray[np.float64] = butter_result[1]  # type: ignore[assignment]
    
    base = os.path.splitext(os.path.basename(eeg_epochs))[0]
    workspace = os.getcwd()
    out_folder = os.path.join(workspace, f"{base}_plv")
    os.makedirs(out_folder, exist_ok=True)
    
    conditions = sorted(eeg_df['condition'].unique().to_list())
    print(f"[PLV] Processing {len(conditions)} conditions: {conditions}")
    
    # Process each condition
    for idx, cond in enumerate(conditions):
        eeg_cond = eeg_df.filter(pl.col('condition') == cond)
        hrv_cond = hrv_df.filter(pl.col('condition') == cond)
        
        epoch_ids = sorted(eeg_cond['epoch_id'].unique().to_list())
        
        plv_results = []
        for ch in eeg_channels:
            ch_plvs = []
            
            for eid in epoch_ids:
                # Get EEG signal and filter
                eeg_signal: NDArray[np.float64] = eeg_cond.filter(pl.col('epoch_id') == eid)[ch].to_numpy()
                eeg_filtered: NDArray[np.float64] = filtfilt(b, a, eeg_signal)  # type: ignore[assignment]
                
                # Get instantaneous phase via Hilbert
                eeg_analytic: NDArray[np.complex128] = hilbert(eeg_filtered)  # type: ignore[assignment]
                eeg_phase: NDArray[np.floating[Any]] = np.angle(eeg_analytic)
                
                # Get HRV R-peak times directly (already in time domain)
                hrv_epoch = hrv_cond.filter(pl.col('epoch_id') == eid)
                if 'R_Peak_Sample' in hrv_epoch.columns or 'time' in hrv_epoch.columns:
                    # Use time column directly - it contains R-peak times
                    rpeak_times: NDArray[np.float64] = hrv_epoch['time'].to_numpy()
                    
                    # Create HRV phase signal (2*pi phase jumps at each R-peak)
                    time_axis: NDArray[np.float64] = eeg_cond.filter(pl.col('epoch_id') == eid)['time'].to_numpy()
                    hrv_phase: NDArray[np.float64] = np.zeros_like(time_axis)
                    
                    for i, t in enumerate(time_axis):
                        # Count number of R-peaks before this time
                        n_rpeaks: int = int(np.sum(rpeak_times <= t))
                        # Phase increases linearly, jumps 2*pi at each R-peak
                        if n_rpeaks > 0 and n_rpeaks < len(rpeak_times):
                            prev_rpeak: np.float64 = rpeak_times[n_rpeaks-1]
                            next_rpeak: np.float64 = rpeak_times[n_rpeaks]
                            phase_frac: np.float64 = (t - prev_rpeak) / (next_rpeak - prev_rpeak)
                            hrv_phase[i] = 2 * np.pi * (n_rpeaks + phase_frac)
                    
                    # Calculate PLV between EEG and HRV phases
                    phase_diff: NDArray[np.floating[Any]] = eeg_phase - hrv_phase
                    plv: float = float(np.abs(np.mean(np.exp(1j * phase_diff))))
                    ch_plvs.append(plv)
            
            if ch_plvs:
                plv_results.append({
                    'channel': ch,
                    'plv_mean': float(np.mean(ch_plvs)),
                    'plv_sem': float(np.std(ch_plvs, ddof=1) / np.sqrt(len(ch_plvs)))
                })
        
        # Output format for plotter
        if plv_results:
            result_df = pl.DataFrame(plv_results)
            output = pl.DataFrame({
                'condition': [cond],
                'x_data': [result_df['channel'].to_list()],
                'y_data': [result_df['plv_mean'].to_list()],
                'y_var': [result_df['plv_sem'].to_list()],
                'plot_type': ['bar'],
                'x_label': ['EEG Channel'],
                'y_label': [f'PLV (EEG-HRV, {freq_band[0]}-{freq_band[1]} Hz)']
            })
            
            out_file = os.path.join(out_folder, f"{base}_plv{idx+1}.parquet")
            output.write_parquet(out_file)
            print(f"[PLV]   {cond}: {os.path.basename(out_file)} ({len(plv_results)} channels)")
    
    # Signal file
    signal_path = os.path.join(workspace, f"{base}_plv.parquet")
    pl.DataFrame({'signal': [1], 'source': [os.path.basename(eeg_epochs)], 'conditions': [len(conditions)]}).write_parquet(signal_path)
    
    print(f"[PLV] Finished. Signal: {os.path.basename(signal_path)}")
    return signal_path

if __name__ == '__main__':
    (lambda a: compute_plv(a[1], a[2], ast.literal_eval(a[3]), ast.literal_eval(a[4])) if len(a) >= 5 else (
        print('[PLV] Usage: python plv_analyzer.py <eeg_epochs.parquet> <hrv_epochs.parquet> <channels_list> <freq_band>'),
        print('[PLV] Example: python plv_analyzer.py eeg_epochs.parquet hrv_epochs.parquet "[\'F3\', \'F4\', \'Fz\']" "(8, 13)"'),
        sys.exit(1)))(sys.argv)
