import polars as pl, sys

def concat_generic(files: list[str], conds: list[str]) -> pl.DataFrame:
    """
    Generic concatenation: Collects all incoming datasets with the same structure.
    
    List fields (x_data, y_data, y_var, counts_per_x, etc.) are collected into lists of lists.
    Metadata fields (plot_type, x_axis, y_label, x_label, y_ticks, etc.) are taken from first dataset.
    Adds 'labels' field containing the list of dataset labels.
    """
    all_dfs = [pl.read_parquet(f).to_dicts()[0] for f in files]
    first_row = all_dfs[0]
    
    list_fields = [k for k, v in first_row.items() if isinstance(v, (list, tuple))]
    metadata_fields = {k: v for k, v in first_row.items() if not isinstance(v, (list, tuple))}
    
    aggregated = {field: [row[field] for row in all_dfs] for field in list_fields}
    aggregated['labels'] = conds
    
    return pl.DataFrame([{**metadata_fields, **aggregated}])

if __name__ == '__main__':
    (lambda a: (lambda pairs: concat_generic([p.split(':')[1] for p in pairs], [p.split(':')[0] for p in pairs]).write_parquet(f"{a[1]}.parquet"))([arg for arg in a[2:]]))(sys.argv)
