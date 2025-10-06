import polars as pl, numpy as np, sys, os, ast
from scipy.signal import butter, filtfilt, hilbert
if __name__ == "__main__":
    usage = lambda: print("Usage: python plv_analyzer.py <signal_parquet_list> <bands_config> <channels_list>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_plv.parquet"
    run = lambda signal_parquet_list, bands_config, channels_list: (
        print(f"[Nextflow] PLV analysis started for input: {signal_parquet_list[0]}") or
        (lambda dfs:
            (lambda results:
                (pl.DataFrame(results).write_parquet(get_output_filename(signal_parquet_list[0])),
                 print(f"[Nextflow] PLV analysis finished for input: {signal_parquet_list[0]}"))
            )([
                dict(
                    band=band,
                    modality_pair=f'{i}-{j}',
                    channel_pair=f'{ch1}-{ch2}',
                    plv_value=(
                        (lambda vals1, vals2, fs1, fs2, band:
                            float(np.abs(np.mean(np.exp(1j * (
                                (
                                    np.angle((lambda sig, fs, band:
                                        (
                                            np.asarray(
                                                hilbert(
                                                    (lambda s, f, b:
                                                        (lambda ba:
                                                            filtfilt(ba[0], ba[1], np.asarray(s, dtype=np.float64)) if ba is not None else np.asarray(s, dtype=np.float64)
                                                        )(butter(2, [b[0]/(f/2), b[1]/(f/2)], btype='band', output='ba') if f > 0 and b[0] > 0 and b[1] > b[0] and b[1] < f/2 else None)
                                                    )(sig, fs, band)
                                                ), dtype=np.complex128
                                            ).flatten() if sig is not None and len(sig) > 0 and fs > 0 and band[0] > 0 and band[1] > band[0] and band[1] < fs/2
                                            else np.asarray(hilbert(np.asarray(sig, dtype=np.float64)), dtype=np.complex128).flatten() if sig is not None and len(sig) > 0 else np.zeros(1, dtype=np.complex128)
                                        ) if sig is not None and len(sig) > 0 else np.zeros(1, dtype=np.complex128)
                                    )(vals1, fs1, band) if vals1 is not None and len(vals1) > 0 else np.zeros(1, dtype=np.complex128)
                                ))
                                -
                                (
                                    np.angle((lambda sig, fs, band:
                                        (
                                            np.asarray(
                                                hilbert(
                                                    (lambda s, f, b:
                                                        (lambda ba:
                                                            filtfilt(ba[0], ba[1], np.asarray(s, dtype=np.float64)) if ba is not None else np.asarray(s, dtype=np.float64)
                                                        )(butter(2, [b[0]/(f/2), b[1]/(f/2)], btype='band', output='ba') if f > 0 and b[0] > 0 and b[1] > b[0] and b[1] < f/2 else None)
                                                    )(sig, fs, band)
                                                ), dtype=np.complex128
                                            ).flatten() if sig is not None and len(sig) > 0 and fs > 0 and band[0] > 0 and band[1] > band[0] and band[1] < fs/2
                                            else np.asarray(hilbert(np.asarray(sig, dtype=np.float64)), dtype=np.complex128).flatten() if sig is not None and len(sig) > 0 else np.zeros(1, dtype=np.complex128)
                                        ) if sig is not None and len(sig) > 0 else np.zeros(1, dtype=np.complex128)
                                    )(vals2, fs2, band) if vals2 is not None and len(vals2) > 0 else np.zeros(1, dtype=np.complex128)
                                ))
                            )))))
                        )(
                            (lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[i], ch1, cond),
                            (lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[j], ch2, cond),
                            (lambda df, ch, cond: 1/(df[(df['channel']==ch)&(df['condition']==cond)]['time'].diff().mean()) if len(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy())>1 and df[(df['channel']==ch)&(df['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0)(dfs[i], ch1, cond),
                            (lambda df, ch, cond: 1/(df[(df['channel']==ch)&(df['condition']==cond)]['time'].diff().mean()) if len(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy())>1 and df[(df['channel']==ch)&(df['condition']==cond)]['time'].diff().mean() != 0.0 else 1.0)(dfs[j], ch2, cond),
                            band
                        )
                    )
                )
                for idx, (i, j) in enumerate([(i, j) for i in range(len(signal_parquet_list)) for j in range(i+1, len(signal_parquet_list))])
                for band, freq in list(bands_config.items())
                for cond in set((lambda dfs, i, j: set(dfs[i]['condition']).intersection(dfs[j]['condition']))(dfs, idx//len(signal_parquet_list), idx%len(signal_parquet_list)))
                for ch1 in channels_list[idx//len(signal_parquet_list)]
                for ch2 in channels_list[idx%len(signal_parquet_list)]
                if len((lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[idx//len(signal_parquet_list)], ch1, cond))>0 and len((lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[idx%len(signal_parquet_list)], ch2, cond))>0 and len((lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[idx//len(signal_parquet_list)], ch1, cond))==len((lambda df, ch, cond: np.asarray(df[(df['channel']==ch)&(df['condition']==cond)]['value'].to_numpy(), dtype=np.float64).flatten() if ch in df['channel'].values and cond in df['condition'].values else np.zeros(1, dtype=np.float64))(dfs[idx%len(signal_parquet_list)], ch2, cond))
            ])
        )([pl.read_parquet(f).to_pandas() for f in signal_parquet_list])
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            signal_parquet_list = ast.literal_eval(args[1])
            bands_config = ast.literal_eval(args[2])
            channels_list = ast.literal_eval(args[3])
            run(signal_parquet_list, bands_config, channels_list)
    except Exception as e:
        print(f"[Nextflow] PLV analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}"); sys.exit(1)
