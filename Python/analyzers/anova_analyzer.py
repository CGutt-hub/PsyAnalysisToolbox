import polars as pl, pingouin as pg, sys, os
if __name__ == "__main__":
    # Print usage and exit if arguments are missing
    usage = lambda: print("Usage: python anova_analyzer.py <input_parquet> <dv> <between> <participant_id> [apply_fdr]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_anova.parquet"
    run = lambda input_parquet, dv, between, participant_id, apply_fdr: (
        print(f"[Nextflow] ANOVA analysis started for participant: {participant_id}"),
        (lambda df: (
            print(f"[Nextflow] Data loaded for ANOVA: shape={df.shape}"),
            (lambda results: (
                print(f"[Nextflow] ANOVA results calculated."),
                (lambda d: (
                    print(f"[Nextflow] FDR correction {'applied' if apply_fdr else 'skipped'}."),
                    (lambda df_out: (
                        print(f"[Nextflow] Writing ANOVA output for participant: {participant_id}"),
                        # Add standardized plotting metadata
                        df_out[1].with_columns([
                            pl.lit("bar").alias("plot_type"),  # ANOVA results -> bar chart
                            pl.lit("ordinal").alias("x_scale"),  # factor levels
                            pl.lit("nominal").alias("y_scale"),  # F-statistics or p-values
                            pl.col("Source").alias("x_data"),  # ANOVA source/factor names
                            pl.col("F").alias("y_data"),  # F-statistics for plotting
                            pl.lit("F-statistic").alias("y_label"),
                            pl.lit(1).alias("plot_weight")
                        ]).write_parquet(get_output_filename(input_parquet)),
                        print(f"[Nextflow] ANOVA analysis finished for participant: {participant_id}")
                    ))(d)
                ))((
                    (lambda fdr: (
                        print(f"[Nextflow] FDR correction results: rejected={fdr[0]}, p_fdr={fdr[1]}"),
                        results.with_columns([
                            pl.Series("p_fdr", fdr[1]),
                            pl.Series("rejected", fdr[0])
                        ])
                    ))(__import__('statsmodels.stats.multitest').stats.multitest.fdrcorrection(results['p-unc'].to_numpy()))
                ) if apply_fdr else results)
            ))(pl.DataFrame(pg.anova(data=df, dv=dv, between=between, detailed=True)))
        ))(pl.read_parquet(input_parquet).to_pandas())
    )
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        else:
            apply_fdr = (len(args) > 5 and args[5].lower() in ["1", "true", "yes"])
            run(args[1], args[2], args[3], args[4], apply_fdr)
    except Exception as e:
        print(f"[Nextflow] ANOVA analysis errored for participant: {sys.argv[4] if len(sys.argv)>4 else 'UNKNOWN'}. Error: {e}"); sys.exit(1)