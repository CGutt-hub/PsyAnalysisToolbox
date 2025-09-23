
import polars as pl, mne, numpy as np, sys

if __name__ == "__main__":
    usage = lambda: print("Usage: python erp_analyzer.py <input_fif> <participant_id>") or sys.exit(1)
    run = lambda input_fif, participant_id, output_parquet: (
        print(f"[Nextflow] ERP analysis started for participant: {participant_id}") or (
            # Lambda: read MNE Epochs object from .fif file
            (lambda epochs:
                # Lambda: flatten nested lists/objects for robust type handling
                (lambda flatten:
                    # Lambda: extract ERP results for each condition/channel
                    (lambda erp_results:
                        # Lambda: convert results to Polars DataFrame and write to Parquet
                        (pl.DataFrame(erp_results).write_parquet(output_parquet),
                         print(f"[Nextflow] ERP analysis finished for participant: {participant_id}"))
                    )([
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
                )(lambda obj, n: next((o for _ in range(n) for o in [obj[0] if isinstance(obj, list) and len(obj) > 0 else obj] if not (isinstance(o, list) and len(o) > 0)), obj))
            )(mne.read_epochs(input_fif, preload=True))
        )
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_fif, participant_id = args[1], args[2]
            output_parquet = f"{participant_id}_erp.parquet"
            run(input_fif, participant_id, output_parquet)
    except Exception as e:
        pid = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        print(f"[Nextflow] ERP analysis errored for participant: {pid}. Error: {e}")
        sys.exit(1)