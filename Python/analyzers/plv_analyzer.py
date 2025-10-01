import polars as pl, numpy as np, sys, ast
from scipy.signal import butter, filtfilt, hilbert
if __name__ == "__main__":
    usage = lambda: print("Usage: python plv_analyzer.py <signal_parquet_list> <bands_config> <channels_list> <participant_id> [output_parquet]") or sys.exit(1)
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        signal_parquet_list = ast.literal_eval(args[1])
        bands_config = ast.literal_eval(args[2])
        channels_list = ast.literal_eval(args[3])
        participant_id = args[4]
        output_parquet = args[5] if len(args) > 5 else f"{participant_id}_plv.parquet"
        # orchestrates PLV computation and output
        print(f"[Nextflow] PLV analysis started for participant: {participant_id}")
        # --- PLV Analysis Orchestration ---
        # 1. Generate all modality pairs (all unique combinations of input signals)
        modality_pairs = [(i, j) for i in range(len(signal_parquet_list)) for j in range(i+1, len(signal_parquet_list))]
        # 2. Read all input signals into pandas DataFrames
        dfs = [pl.read_parquet(f).to_pandas() for f in signal_parquet_list]
        # 3. Generate channel pairs for each modality pair
        channel_pairs = [(channels_list[i], channels_list[j]) for i, j in modality_pairs]
        # 4. Prepare frequency band items
        band_items = list(bands_config.items())

        results = []
        for idx, (i, j) in enumerate(modality_pairs):
            df1, df2 = dfs[i], dfs[j]
            channels1, channels2 = channel_pairs[idx]
            for band, freq in band_items:
                for cond in set(df1['condition']).intersection(df2['condition']):
                    for ch1 in channels1:
                        for ch2 in channels2:
                            vals1 = df1[(df1['channel']==ch1)&(df1['condition']==cond)]['value'].to_numpy()
                            vals2 = df2[(df2['channel']==ch2)&(df2['condition']==cond)]['value'].to_numpy()
                            if len(vals1)>0 and len(vals2)>0 and len(vals1)==len(vals2):
                                fs1 = 1/(df1[(df1['channel']==ch1)&(df1['condition']==cond)]['time'].diff().mean()) if len(vals1)>1 and df1[(df1['channel']==ch1)&(df1['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0
                                fs2 = 1/(df2[(df2['channel']==ch2)&(df2['condition']==cond)]['time'].diff().mean()) if len(vals2)>1 and df2[(df2['channel']==ch2)&(df2['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0
                                # --- Nested lambda for PLV calculation ---
                                def bandpass_and_hilbert(sig, fs, band):
                                    if fs <= 0 or band[0] <= 0 or band[1] <= band[0] or band[1] >= fs/2:
                                        print(f"[PLV] Invalid band or fs: band={band}, fs={fs}")
                                        return None
                                    butter_out = butter(2, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='ba')
                                    if butter_out is None or not isinstance(butter_out, (tuple, list)) or len(butter_out) < 2:
                                        print(f"[PLV] Butter returned invalid output: {butter_out}")
                                        return None
                                    b, a = butter_out[:2]
                                    try:
                                        filtered = filtfilt(b, a, np.asarray(sig, dtype=np.float64))
                                        analytic = hilbert(filtered)
                                        return np.asarray(analytic)
                                    except Exception as e:
                                        print(f"[PLV] Filtering/hilbert error: {e}")
                                        return None

                                plv_calc = lambda x, y, fs_x, fs_y, band: (
                                    float(np.abs(np.mean(np.exp(1j * (
                                        (np.angle(np.asarray(bandpass_and_hilbert(x, fs_x, band))) if bandpass_and_hilbert(x, fs_x, band) is not None else 0) -
                                        (np.angle(np.asarray(bandpass_and_hilbert(y, fs_y, band))) if bandpass_and_hilbert(y, fs_y, band) is not None else 0)
                                    )))))
                                )
                                plv_value = plv_calc(vals1, vals2, fs1, fs2, band)
                                # --- Collect results ---
                                results.append(dict(
                                    participant_id=participant_id,
                                    band=band,
                                    modality_pair=f'{i}-{j}',
                                    channel_pair=f'{ch1}-{ch2}',
                                    plv_value=plv_value
                                ))
        # --- Write results to output ---
        pl.DataFrame(results).write_parquet(output_parquet)
    except Exception as e:
        pid = sys.argv[4] if len(sys.argv) > 4 else "unknown"
        print(f"[Nextflow] PLV analysis errored for participant: {pid}. Error: {e}")
        sys.exit(1)
