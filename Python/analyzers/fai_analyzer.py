import polars as pl, numpy as np, sys, ast

if __name__ == "__main__":
    usage = lambda: print("Usage: python fai_analyzer.py <input_parquet> <fai_band_name> <electrode_pairs> <participant_id>\nelectrode_pairs format: [(left1,right1),(left2,right2)] (as a Python list string)") or sys.exit(1)
    run = lambda input_parquet, fai_band_name, electrode_pairs, participant_id, output_parquet: (
        print(f"[Nextflow] FAI analysis started for participant: {participant_id}"),
        (lambda df: (
            print(f"[Nextflow] Data loaded for FAI: shape={df.shape}"),
            (lambda results: (
                print(f"[Nextflow] FAI results calculated: {len(results)} entries."),
                (lambda _: (
                    print(f"[Nextflow] Writing FAI output for participant: {participant_id}"),
                    results.write_parquet(output_parquet),
                    print(f"[Nextflow] FAI analysis finished for participant: {participant_id}")
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
        if len(args) < 5:
            usage()
        else:
            input_parquet = args[1]
            fai_band_name = args[2]
            electrode_pairs = ast.literal_eval(args[3])
            participant_id = args[4]
            output_parquet = f"{participant_id}_fai.parquet"
            run(input_parquet, fai_band_name, electrode_pairs, participant_id, output_parquet)
    except Exception as e:
        pid = sys.argv[4] if len(sys.argv) > 4 else "unknown"
        print(f"[Nextflow] FAI analysis errored for participant: {pid}. Error: {e}")
        sys.exit(1)