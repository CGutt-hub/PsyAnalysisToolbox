import polars as pl, sys
from scipy.stats import pearsonr
if __name__ == "__main__":
    # Print usage and exit if arguments are missing
    usage = lambda: print("Usage: python correlation_analyzer.py <input_parquet> <participant_id>") or sys.exit(1)
    run = lambda input_parquet, participant_id: (
        print(f"[Nextflow] Correlation analysis started for participant: {participant_id}"),
        (lambda df: (
            print(f"[Nextflow] Data loaded for correlation: shape={df.shape}"),
            (lambda num: (
                print(f"[Nextflow] Numeric columns selected: {num}"),
                (lambda results: (
                    print(f"[Nextflow] Pairwise Pearson correlations calculated."),
                    print(f"[Nextflow] Writing correlation output for participant: {participant_id}"),
                    results.write_parquet(f"{participant_id}_correlation.parquet"),
                    print(f"[Nextflow] Correlation analysis finished for participant: {participant_id}")
                ))(pl.DataFrame([
                    {
                        'var1': c1,
                        'var2': c2,
                        'correlation': pearsonr(df[c1].to_numpy(), df[c2].to_numpy())[0],
                        'p': pearsonr(df[c1].to_numpy(), df[c2].to_numpy())[1]
                    }
                    for i, c1 in enumerate(num) for c2 in num[i+1:]
                ]))
            ))(df.select(pl.NUMERIC_DTYPES).columns)
        ))(pl.read_parquet(input_parquet))
    )
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            run(args[1], args[2])
    except Exception as e:
        print(f"[Nextflow] Correlation analysis errored for participant: {sys.argv[2] if len(sys.argv)>2 else 'UNKNOWN'}. Error: {e}"); sys.exit(1)