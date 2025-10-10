import polars as pl, sys, os
from scipy.stats import pearsonr
if __name__ == "__main__":
    usage = lambda: print("Usage: python correlation_analyzer.py <input_parquet>") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_correl.parquet"
    run = lambda input_parquet: (
        print(f"[Nextflow] Correlation analysis started for input: {input_parquet}"),
        (lambda df: (
            print(f"[Nextflow] Data loaded for correlation: shape={df.shape}"),
            (lambda num: (
                print(f"[Nextflow] Numeric columns selected: {num}"),
                (lambda results: (
                    print(f"[Nextflow] Pairwise Pearson correlations calculated."),
                    print(f"[Nextflow] Writing correlation output for input: {input_parquet}"),
                    results.write_parquet(get_output_filename(input_parquet)),
                    print(f"[Nextflow] Correlation analysis finished for input: {input_parquet}")
                ))(pl.DataFrame([
                    {
                        # Original analysis data
                        'var1': c1,
                        'var2': c2,
                        'correlation': pearsonr(df[c1].to_numpy(), df[c2].to_numpy())[0],
                        'p': pearsonr(df[c1].to_numpy(), df[c2].to_numpy())[1],
                        # Standardized plotting metadata
                        'plot_type': 'scatter',
                        'x_scale': 'nominal', 
                        'y_scale': 'nominal',
                        'x_data': f"{c1}_vs_{c2}",
                        'y_data': pearsonr(df[c1].to_numpy(), df[c2].to_numpy())[0],
                        'y_label': 'Correlation (r)', 'plot_weight': 1
                    }
                    for i, c1 in enumerate(num) for c2 in num[i+1:]
                ]))
            ))(df.select(pl.NUMERIC_DTYPES).columns)
        ))(pl.read_parquet(input_parquet))
    )
    try:
        args = sys.argv
        if len(args) < 2:
            usage()
        else:
            run(args[1])
    except Exception as e:
        print(f"[Nextflow] Correlation analysis errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}"); sys.exit(1)