
import polars as pl, mne, numpy as np, sys, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python erp_analyzer.py <input_fif>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_erp.parquet"
    run = lambda input_fif: (
        print(f"[Nextflow] ERP analysis started for input: {input_fif}"),
        (lambda epochs: (
            print(f"[Nextflow] MNE Epochs object loaded."),
            (lambda flatten: (
                print(f"[Nextflow] Flatten function ready."),
                (lambda erp_results: (
                    print(f"[Nextflow] ERP results extracted: {len(erp_results)} entries."),
                    print(f"[Nextflow] Writing ERP output for input: {input_fif}"),
                    pl.DataFrame(erp_results).write_parquet(get_output_filename(input_fif)),
                    print(f"[Nextflow] ERP analysis finished for input: {input_fif}")
                ))([
                    {
                        'condition': cond,
                        'channel': ch_name,
                        'latency': evoked.times[peak_idx],
                        'amplitude': evoked.data[ch_idx][peak_idx]
                    }
                    for cond in epochs.event_id
                    for cond_epochs in [flatten(epochs[cond], 10)]
                    if isinstance(cond_epochs, mne.Epochs)
                    for evoked in [flatten(cond_epochs.average(), 2)]
                    if isinstance(evoked, mne.Evoked)
                    for ch_idx, ch_name in enumerate(evoked.ch_names)
                    for peak_idx in [np.argmax(np.abs(evoked.data[ch_idx]))]
                ])
            ))(lambda obj, n: next((o for _ in range(n) for o in [obj[0] if isinstance(obj, list) and len(obj) > 0 else obj] if not (isinstance(o, list) and len(o) > 0)), obj))
        ))(mne.read_epochs(input_fif, preload=True))
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            input_fif = args[1]
            run(input_fif)
    except Exception as e:
        print(f"[Nextflow] ERP analysis errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)