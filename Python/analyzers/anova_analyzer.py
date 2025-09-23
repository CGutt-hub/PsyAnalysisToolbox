import polars as pl, pingouin as pg, sys
if __name__ == "__main__":
    # Print usage and exit if arguments are missing
    usage = lambda: print("Usage: python anova_analyzer.py <input_parquet> <dv> <between> <participant_id> [apply_fdr]") or sys.exit(1)
    run = lambda input_parquet, dv, between, participant_id, apply_fdr: (
        print(f"[Nextflow] ANOVA analysis started for participant: {participant_id}") or [
            # Read input data using Polars and convert to pandas for Pingouin
            (
                lambda df: [
                    # Lambda-driven ANOVA using Pingouin
                    (
                        lambda results: [
                            # Optional FDR correction step
                            (
                                lambda df_out: [
                                    df_out.write_parquet(f"{participant_id}_anova.parquet"),
                                    print(f"[Nextflow] ANOVA analysis finished for participant: {participant_id}")
                                ][-1]
                            )(
                                (lambda d: (
                                    # If FDR requested, add corrected p-values and rejection mask
                                    (lambda fdr:
                                        d.with_columns([
                                            pl.Series("p_fdr", fdr[1]),
                                            pl.Series("rejected", fdr[0])
                                        ])
                                    )( __import__('statsmodels.stats.multitest').stats.multitest.fdrcorrection(d['p-unc'].to_numpy()) )
                                ) if apply_fdr else d
                                )(results)
                            )
                        ][-1]
                    )(
                        pl.DataFrame(
                            pg.anova(data=df, dv=dv, between=between, detailed=True)
                        )
                    )
                ][-1]
            )(pl.read_parquet(input_parquet).to_pandas())
        ][-1]
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