
import polars as pl, pandas as pd, mne, sys, ast
from mne_nirs.statistics import run_glm

if __name__ == "__main__":
    # Print usage and exit if arguments are missing
    usage = lambda: print("Usage: python glm_analyzer.py <input_parquet> <design_matrix_parquet> <sfreq> <ch_types_map> <contrasts_config> <participant_id>\nch_types_map and contrasts_config should be Python dict strings.") or sys.exit(1)
    # Main GLM analysis procedure (ultra-compressed, lambda-driven)
    run = lambda input_parquet, design_matrix_parquet, sfreq, ch_types_map, contrasts_config, participant_id: (
        print(f"[Nextflow] GLM analysis started for participant: {participant_id}") or [
            # Read input data using Polars, convert to pandas for MNE compatibility
            (
                lambda df: [
                    # Extract and sort channel names
                    (
                        lambda ch_names: [
                            # Map channel types from config
                            (
                                lambda ch_types_list: [
                                    # Use single string if all types match, else first type (MNE limitation)
                                    (
                                        lambda ch_types: [
                                            # Create MNE RawArray for GLM
                                            (
                                                lambda raw: [
                                                    # Run GLM analysis using MNE-NIRS
                                                    (
                                                        lambda glm_results: [
                                                            # Extract contrasts if provided, else use all results
                                                            (
                                                                lambda final_df: [
                                                                    # Write results to Parquet, print completion
                                                                    pl.DataFrame(final_df if not final_df.empty else []).write_parquet(f"{participant_id}_glm.parquet"),
                                                                    print(f"[Nextflow] GLM analysis finished for participant: {participant_id}")
                                                                ][-1]
                                                            )(
                                                                pd.concat([
                                                                    df_.assign(contrast=name)
                                                                    for name, weights in contrasts_config.items()
                                                                    for df_ in [glm_results.compute_contrast(weights).to_dataframe()]
                                                                    if not df_.empty
                                                                ], ignore_index=True) if contrasts_config else glm_results.results.to_dataframe()
                                                            )
                                                        ][-1]
                                                    )(run_glm(raw, pl.read_parquet(design_matrix_parquet).to_pandas()))
                                                ][-1]
                                            )(mne.io.RawArray(
                                                df.pivot_table(index='channel', columns='time', values='value').reindex(ch_names).to_numpy(),
                                                mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types if isinstance(ch_types, str) else ch_types_list[0]),
                                                verbose=False
                                            ))
                                        ][-1]
                                    )(ch_types_list[0] if all(ct == ch_types_list[0] for ct in ch_types_list) else ch_types_list[0])
                                ][-1]
                            )([str(ch_types_map.get(ch, next(iter(ch_types_map.values())))) for ch in ch_names])
                        ][-1]
                    )(sorted(df['channel'].unique().tolist()))
                ][-1]
            )(pl.read_parquet(input_parquet).to_pandas())
        ][-1]
    )
    try:
        args = sys.argv
        if len(args) < 7:
            usage()
        else:
            run(args[1], args[2], float(args[3]), ast.literal_eval(args[4]), ast.literal_eval(args[5]), args[6])
    except Exception as e:
        print(f"[Nextflow] GLM analysis errored. Error: {e}")
        sys.exit(1)