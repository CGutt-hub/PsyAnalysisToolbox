import polars as pl, numpy as np, sys, ast, os
if __name__ == "__main__":
    usage = lambda: print("Usage: python fai_analyzer.py <input_parquet> <fai_band_name> <electrode_pairs>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_fai.parquet"
    run = lambda input_parquet, fai_band_name, electrode_pairs: (
        print(f"[FAI] FAI analysis started for input: {input_parquet}"),
        (lambda df: (
            print(f"[FAI] Data loaded for FAI: shape={df.shape}"),
            (lambda results: (
                print(f"[FAI] FAI results calculated: {len(results)} entries."),
                (lambda _: (
                    print(f"[FAI] Writing FAI output for input: {input_parquet}"),
                    results.write_parquet(get_output_filename(input_parquet)),
                    print(f"[FAI] FAI analysis finished for input: {input_parquet}")
                ))(results)
            ))(pl.DataFrame([
                {
                    'condition': cond,
                    'band': fai_band_name,
                    'fai_value': float(
                        np.log(df.filter((df['channel'] == right) & (df['band'] == fai_band_name) & (df['condition'] == cond))['power'].to_numpy()[0])
                        - np.log(df.filter((df['channel'] == left) & (df['band'] == fai_band_name) & (df['condition'] == cond))['power'].to_numpy()[0])
                    )
                }
                for left, right in electrode_pairs
                for cond in df['condition'].unique()
                if (
                    len(df.filter((df['channel'] == left) & (df['band'] == fai_band_name) & (df['condition'] == cond))['power']) > 0 and
                    len(df.filter((df['channel'] == right) & (df['band'] == fai_band_name) & (df['condition'] == cond))['power']) > 0
                )
            ]) if df is not None and len(df) > 0 else pl.DataFrame([]))
        ))(pl.read_parquet(input_parquet))
    )
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            input_parquet = args[1]
            fai_band_name = args[2]
            electrode_pairs = ast.literal_eval(args[3])
            run(input_parquet, fai_band_name, electrode_pairs)
    except Exception as e:
        print(f"[FAI] FAI analysis errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)