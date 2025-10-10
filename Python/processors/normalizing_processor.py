import polars as pl, numpy as np, sys, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

if __name__ == "__main__":
    usage = lambda: print("Usage: python normalization_processor.py <input_parquet> <normalization_type> <data_columns> [group_by_columns]") or sys.exit(1)
    get_output_filename = lambda input_file: f"{os.path.splitext(os.path.basename(input_file))[0]}_normalized.parquet"
    
    run = lambda input_parquet, normalization_type, data_columns, group_by_columns: (
        print(f"[Nextflow] Generic normalization started for: {input_parquet}") or
        (lambda df:
            (lambda column_list:
                (lambda group_cols:
                    (lambda normalized_df:
                        normalized_df.write_parquet(get_output_filename(input_parquet)) or
                        print(f"[Nextflow] Generic normalization finished. Output: {get_output_filename(input_parquet)}")
                    )(
                        # Apply normalization to specified columns
                        df.with_columns([
                            (lambda norm_type:
                                # Z-score normalization
                                (pl.col(col) - pl.col(col).mean()) / pl.col(col).std() if norm_type == 'zscore' else
                                # Min-Max normalization 
                                (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()) if norm_type == 'minmax' else
                                # Robust normalization (median and MAD)
                                (pl.col(col) - pl.col(col).median()) / (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)) if norm_type == 'robust' else
                                # Log transformation
                                pl.col(col).log() if norm_type == 'log' else
                                # Unit vector normalization
                                pl.col(col) / pl.col(col).map_elements(lambda x: np.sqrt(np.sum(np.array(x)**2)), return_dtype=pl.Float64) if norm_type == 'unit' else
                                # Default: return original
                                pl.col(col)
                            )(normalization_type.lower()).alias(f"{col}_normalized")
                            for col in column_list
                        ])
                    )
                )(group_by_columns.split(',') if group_by_columns and group_by_columns.strip() else [])
            )(data_columns.split(',') if isinstance(data_columns, str) else data_columns)
        )(pl.read_parquet(input_parquet))
    )
    
    try:
        args = sys.argv
        if len(args) < 4:
            usage()
        else:
            input_parquet, normalization_type, data_columns = args[1], args[2], args[3]
            group_by_columns = args[4] if len(args) > 4 else ""
            run(input_parquet, normalization_type, data_columns, group_by_columns)
    except Exception as e:
        print(f"[Nextflow] Normalizing errored for input: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}. Error: {e}")
        sys.exit(1)