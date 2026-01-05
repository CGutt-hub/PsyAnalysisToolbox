
import polars as pl, pandas as pd, mne, sys, ast, os
from mne_nirs.statistics import run_glm
if __name__ == "__main__":
    usage = lambda: print("Usage: python glm_analyzer.py <input_parquet> <sfreq> <ch_types_map> <contrasts_config>") or sys.exit(1)
    
    def run(input_parquet, sfreq, ch_types_map, contrasts_config):
        print(f"[GLM] GLM analysis started for input: {input_parquet}")
        base = os.path.splitext(os.path.basename(input_parquet))[0]
        workspace = os.getcwd()
        out_folder = os.path.join(workspace, f"{base}_glm")
        os.makedirs(out_folder, exist_ok=True)
        
        print(f"[GLM] Loading input epochs: {input_parquet}")
        df = pl.read_parquet(input_parquet).to_pandas()
        print(f"[GLM] Loaded epochs DataFrame shape: {df.shape}")
        
        ch_names = sorted(set(df['channel']))
        ch_types_list = list(ch_types_map.values())
        ch_types = ch_types_list[0] if all(t == ch_types_list[0] for t in ch_types_list) else ch_types_list
        
        print(f"[GLM] Channel names: {ch_names}")
        print(f"[GLM] Using channel type(s): {ch_types}")
        
        raw = mne.io.RawArray(
            df.pivot_table(index='channel', columns='time', values='value').reindex(ch_names).to_numpy(),
            mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types if isinstance(ch_types, str) else ch_types_list[0]),
            verbose=False
        )
        print(f"[GLM] Created MNE RawArray for GLM.")
        
        glm_results = run_glm(raw, None)
        print(f"[GLM] GLM results computed.")
        
        if contrasts_config:
            contrasts = list(contrasts_config.keys())
            print(f"[GLM] Processing {len(contrasts)} contrasts: {contrasts}")
            
            for idx, (name, weights) in enumerate(contrasts_config.items()):
                contrast_df = glm_results.compute_contrast(weights).to_dataframe()
                if not contrast_df.empty:
                    result = pl.DataFrame(contrast_df).with_columns([
                        pl.lit("bar").alias("plot_type"),
                        pl.lit("ordinal").alias("x_scale"),
                        pl.lit("nominal").alias("y_scale"),
                        pl.lit(name).alias("x_data"),
                        pl.lit(0.0).alias("y_data"),
                        pl.lit("t-statistic").alias("y_label"),
                        pl.lit(1).alias("plot_weight")
                    ])
                    
                    out_path = os.path.join(out_folder, f"{base}_glm{idx+1}.parquet")
                    result.write_parquet(out_path)
                    print(f"[GLM]   {name}: {os.path.basename(out_path)}")
            
            signal_path = os.path.join(workspace, f"{base}_glm.parquet")
            pl.DataFrame({
                'signal': [1],
                'source': [os.path.basename(input_parquet)],
                'conditions': [len(contrasts)],
                'folder_path': [os.path.abspath(out_folder)]
            }).write_parquet(signal_path)
            print(f"[GLM] GLM analysis finished. Signal: {signal_path}")
        else:
            # No contrasts - single output
            result_df = glm_results.results.to_dataframe()
            out_path = os.path.join(out_folder, f"{base}_glm1.parquet")
            pl.DataFrame(result_df if not result_df.empty else []).write_parquet(out_path)
            
            signal_path = os.path.join(workspace, f"{base}_glm.parquet")
            pl.DataFrame({
                'signal': [1],
                'source': [os.path.basename(input_parquet)],
                'conditions': [1],
                'folder_path': [os.path.abspath(out_folder)]
            }).write_parquet(signal_path)
            print(f"[GLM] GLM analysis finished. Signal: {signal_path}")
    try:
        args = sys.argv
        if len(args) < 5:
            usage()
        else:
            try:
                run(args[1], float(args[2]), ast.literal_eval(args[3]), ast.literal_eval(args[4]))
            except Exception as e:
                print(f"[GLM] GLM analysis errored inside run. Error: {e}")
                sys.exit(1)
    except Exception as e:
        print(f"[GLM] GLM analysis errored. Error: {e}")
        sys.exit(1)