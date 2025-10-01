
import polars as pl, pandas as pd, mne, sys, ast
from mne_nirs.statistics import run_glm

if __name__ == "__main__":
    usage = lambda: print("Usage: python glm_analyzer.py <input_parquet> <design_matrix_parquet> <sfreq> <ch_types_map> <contrasts_config> <participant_id>\nch_types_map and contrasts_config should be Python dict strings.") or sys.exit(1)
    run = lambda input_parquet, design_matrix_parquet, sfreq, ch_types_map, contrasts_config, participant_id: (
        print(f"[Nextflow] GLM analysis started for participant: {participant_id}"),
        print(f"[Nextflow] Loading input epochs: {input_parquet}"),
        print(f"[Nextflow] Loading design matrix: {design_matrix_parquet}"),
        (lambda df: (
            print(f"[Nextflow] Loaded epochs DataFrame shape: {df.shape}"),
            (lambda ch_names: (
                print(f"[Nextflow] Channel names: {ch_names}"),
                (lambda ch_types_list: (
                    print(f"[Nextflow] Channel types: {ch_types_list}"),
                    (lambda ch_types: (
                        print(f"[Nextflow] Using channel type(s): {ch_types}"),
                        (lambda raw: (
                            print(f"[Nextflow] Created MNE RawArray for GLM."),
                            (lambda glm_results: (
                                print(f"[Nextflow] GLM results computed."),
                                (lambda final_df: (
                                    print(f"[Nextflow] Final GLM DataFrame shape: {final_df.shape}"),
                                    pl.DataFrame(final_df if not final_df.empty else []).write_parquet(f"{participant_id}_glm.parquet"),
                                    print(f"[Nextflow] GLM analysis finished for participant: {participant_id}")
                                ))(
                                    pd.concat([
                                        df_.assign(contrast=name)
                                        for name, weights in contrasts_config.items()
                                        for df_ in [glm_results.compute_contrast(weights).to_dataframe()]
                                        if not df_.empty
                                    ], ignore_index=True) if contrasts_config else glm_results.results.to_dataframe()
                                )
                            ))(run_glm(raw, pl.read_parquet(design_matrix_parquet).to_pandas()))
                        ))(mne.io.RawArray(
                            df.pivot_table(index='channel', columns='time', values='value').reindex(ch_names).to_numpy(),
                            mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types if isinstance(ch_types, str) else ch_types_list[0]),
                            verbose=False
                        ))
                    ))(ch_types_list[0] if all(t == ch_types_list[0] for t in ch_types_list) else ch_types_list)
                ))(list(ch_types_map.values()))
            ))(sorted(set(df['channel'])))
        ))(pl.read_parquet(input_parquet).to_pandas())
    )
    try:
        args = sys.argv
        if len(args) < 7:
            usage()
        else:
            try:
                run(args[1], args[2], float(args[3]), ast.literal_eval(args[4]), ast.literal_eval(args[5]), args[6])
            except Exception as e:
                print(f"[Nextflow] GLM analysis errored inside run. Error: {e}")
                sys.exit(1)
    except Exception as e:
        print(f"[Nextflow] GLM analysis errored. Error: {e}")
        sys.exit(1)