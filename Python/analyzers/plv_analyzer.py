import polars as pl, numpy as np, sys, ast
if __name__ == "__main__":
    # Nextflow-compliant CLI: input files, bands config, channel lists, participant ID
    if len(sys.argv) < 7:
        print("Usage: python plv_analyzer.py <signal1_parquet> <signal2_parquet> <bands_config> <channels1> <channels2> <participant_id>")
        sys.exit(1)
    signal1_parquet, signal2_parquet, bands_config, channels1, channels2, participant_id = sys.argv[1], sys.argv[2], ast.literal_eval(sys.argv[3]), ast.literal_eval(sys.argv[4]), ast.literal_eval(sys.argv[5]), sys.argv[6]
    output_parquet = f"{participant_id}_plv.parquet"
    print(f"[Nextflow] PLV analysis started for participant: {participant_id}")
    try:
        # Import signal processing functions
        from scipy.signal import hilbert, butter, filtfilt
        # Lambda for PLV calculation with nested bandpass and hilbert1d
        plv = lambda x, y, fs_x, fs_y, band: (
            # Lambda: defines bandpass and hilbert1d for locality
            (lambda bandpass, hilbert1d:
                # Lambda: calculates PLV using bandpass and Hilbert transform
                float(np.abs(np.mean(np.exp(1j*(
                    np.angle(np.asarray(hilbert1d(bandpass(x, fs_x, band)))) -
                    np.angle(np.asarray(hilbert1d(bandpass(y, fs_y, band))))
                )))))
            )(
                # Lambda: bandpass filter (flattens input, applies Butterworth, filtfilt)
                lambda data, fs, band: (
                    # Lambda: flattens input array
                    (lambda arr:
                        # Lambda: applies Butterworth filter and filtfilt
                        (lambda filt: filtfilt(*filt, arr) if isinstance(filt, tuple) and len(filt) == 2 else arr)
                        (butter(2, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')) if arr.size > 0 else np.zeros(1)
                    )(np.asarray(data).flatten())
                ),
                # Lambda: Hilbert transform (flattens input, applies Hilbert, unpacks tuple if needed)
                lambda x: (
                    # Lambda: flattens input array
                    (lambda arr:
                        # Lambda: unpacks Hilbert output if tuple
                        (lambda h: h[0] if isinstance(h, tuple) else h)(hilbert(arr))
                    )(np.asarray(x).flatten())
                )
            )
        )
        # PLV extraction
        (
            # Lambda: reads input data using Polars and converts to pandas
            lambda df1, df2: [
                # Lambda: builds results DataFrame with PLV values for all valid channel/band/condition pairs
                (
                    # Lambda: writes results to Parquet and prints completion
                    lambda results: [
                        results.write_parquet(output_parquet),
                        print(f"[Nextflow] PLV analysis finished for participant: {participant_id}")
                    ][-1]
                )(
                    pl.DataFrame([
                        {
                            'condition': cond,  # Experimental condition
                            'band': band,        # Frequency band label
                            'modality_pair': f'{ch1}-{ch2}',  # Channel pair
                            # Calculate PLV for this channel pair, band, and condition
                            'plv_value': plv(
                                df1[(df1['channel']==ch1)&(df1['condition']==cond)]['value'].to_numpy(),
                                df2[(df2['channel']==ch2)&(df2['condition']==cond)]['value'].to_numpy(),
                                1/(df1[(df1['channel']==ch1)&(df1['condition']==cond)]['time'].diff().mean()) if len(df1[(df1['channel']==ch1)&(df1['condition']==cond)]['value'])>1 and df1[(df1['channel']==ch1)&(df1['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0,
                                1/(df2[(df2['channel']==ch2)&(df2['condition']==cond)]['time'].diff().mean()) if len(df2[(df2['channel']==ch2)&(df2['condition']==cond)]['value'])>1 and df2[(df2['channel']==ch2)&(df2['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0,
                                freq
                            )
                        }
                        for band, freq in bands_config.items()
                        for cond in set(df1['condition']).intersection(df2['condition'])
                        for ch1 in channels1
                        for ch2 in channels2
                        # Only include valid channel/condition pairs with matching lengths
                        if (
                            len(df1[(df1['channel']==ch1)&(df1['condition']==cond)]['value'])>0 and
                            len(df2[(df2['channel']==ch2)&(df2['condition']==cond)]['value'])>0 and
                            len(df1[(df1['channel']==ch1)&(df1['condition']==cond)]['value'])==len(df2[(df2['channel']==ch2)&(df2['condition']==cond)]['value'])
                        )
                    ])
                )
            ][-1]
        )(pl.read_parquet(signal1_parquet).to_pandas(), pl.read_parquet(signal2_parquet).to_pandas())
    except Exception as e:
        print(f"[Nextflow] PLV analysis errored for participant: {participant_id}. Error: {e}")
        sys.exit(1)