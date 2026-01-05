
import polars as pl, mne, numpy as np, sys, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python erp_analyzer.py <input_fif>") or sys.exit(1)
    
    def run(input_fif):
        print(f"[ERP] ERP analysis started for input: {input_fif}")
        base = os.path.splitext(os.path.basename(input_fif))[0]
        workspace = os.getcwd()
        out_folder = os.path.join(workspace, f"{base}_erp")
        os.makedirs(out_folder, exist_ok=True)
        
        epochs = mne.read_epochs(input_fif, preload=True)
        print(f"[ERP] MNE Epochs object loaded.")
        
        flatten = lambda obj, n: next((o for _ in range(n) for o in [obj[0] if isinstance(obj, list) and len(obj) > 0 else obj] if not (isinstance(o, list) and len(o) > 0)), obj)
        
        conditions = list(epochs.event_id.keys())
        print(f"[ERP] Processing {len(conditions)} conditions: {conditions}")
        
        for idx, cond in enumerate(conditions):
            cond_epochs = flatten(epochs[cond], 10)
            if not isinstance(cond_epochs, mne.Epochs):
                continue
            
            evoked = flatten(cond_epochs.average(), 2)
            if not isinstance(evoked, mne.Evoked):
                continue
            
            erp_results = []
            for ch_idx, ch_name in enumerate(evoked.ch_names):
                peak_idx = np.argmax(np.abs(evoked.data[ch_idx]))
                erp_results.append({
                    'channel': ch_name,
                    'latency': evoked.times[peak_idx],
                    'amplitude': evoked.data[ch_idx][peak_idx],
                    'plot_type': 'line',
                    'x_scale': 'nominal',
                    'y_scale': 'nominal',
                    'x_data': evoked.times[peak_idx],
                    'y_data': evoked.data[ch_idx][peak_idx],
                    'x_label': 'Time (s)',
                    'y_label': 'Amplitude (μV)',
                    'plot_weight': 1
                })
            
            out_path = os.path.join(out_folder, f"{base}_erp{idx+1}.parquet")
            pl.DataFrame(erp_results).write_parquet(out_path)
            print(f"[ERP]   {cond}: {os.path.basename(out_path)} ({len(erp_results)} channels)")
        
        signal_path = os.path.join(workspace, f"{base}_erp.parquet")
        pl.DataFrame({
            'signal': [1],
            'source': [os.path.basename(input_fif)],
            'conditions': [len(conditions)],
            'folder_path': [os.path.abspath(out_folder)]
        }).write_parquet(signal_path)
        print(f"[ERP] ERP analysis finished. Signal: {signal_path}")
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_fif = args[1]
            run(input_fif)
    except Exception as e:
        print(f"[ERP] ERP analysis errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)