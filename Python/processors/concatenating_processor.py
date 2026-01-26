import polars as pl, sys, os, re

def extract_pid(filepath: str) -> str:
    """Extract participant ID from filepath (pattern like EV_002 or P001)."""
    basename = os.path.basename(filepath)
    match = re.match(r'^([A-Za-z]+_\d+)', basename)
    return match.group(1) if match else ''

def concat_generic(files: list[str], conds: list[str]) -> pl.DataFrame:
    """
    Generic concatenation: Collects all incoming datasets with the same structure.
    
    List fields (y_data, y_var, counts_per_x, etc.) are collected into lists of lists.
    For 'grid' or 'bar' plot types, x_data is treated as shared categories (metadata).
    Metadata fields (plot_type, x_axis, y_label, x_label, y_ticks, etc.) are taken from first dataset.
    Adds 'labels' field containing the list of dataset labels.
    Labels are extracted from 'condition' field in each file if available, otherwise use provided conds.
    """
    print(f"[concatenating] Concatenating {len(files)} files")
    all_dfs = [pl.read_parquet(f).to_dicts()[0] for f in files]
    first_row = all_dfs[0]
    
    # Extract labels from 'condition' field if available, otherwise use provided conds
    labels = [row.get('condition', conds[i] if i < len(conds) else f'cond{i+1}') for i, row in enumerate(all_dfs)]
    print(f"[concatenating] Labels extracted: {labels}")
    
    list_fields = [k for k, v in first_row.items() if isinstance(v, (list, tuple))]
    # These list fields are actually metadata (same across all files), not data to aggregate
    metadata_list_fields = ['y_labels']  # Endpoint labels like ['gar nicht', 'extrem']
    
    # For grid/bar plots, x_data represents shared categories (ROI names, question names, etc.)
    # and should NOT be nested - it stays as a flat list
    plot_type = first_row.get('plot_type', '')
    if plot_type in ('grid', 'bar'):
        metadata_list_fields.append('x_data')
    
    list_fields = [k for k in list_fields if k not in metadata_list_fields]
    # Exclude 'condition' from metadata since it varies per file
    metadata_fields = {k: v for k, v in first_row.items() if not isinstance(v, (list, tuple)) and k != 'condition'}
    # Add metadata list fields
    for k in metadata_list_fields:
        if k in first_row:
            metadata_fields[k] = first_row[k]
    
    print(f"[concatenating] List fields (to aggregate): {list_fields}")
    print(f"[concatenating] Metadata fields: {list(metadata_fields.keys())}")
    
    aggregated = {field: [row[field] for row in all_dfs] for field in list_fields}
    aggregated['labels'] = labels
    
    return pl.DataFrame([{**metadata_fields, **aggregated}])

if __name__ == '__main__': (lambda a:
    (lambda items, out_base: (
        (lambda files, labels: (
            (lambda pid, out_path: (
                concat_generic(files, labels).write_parquet(out_path),
                print(f"[concatenating] Concatenated {len(files)} files -> {out_path}"),
                print(out_path)
            ))(extract_pid(files[0]) if files else '', 
               os.path.join(os.getcwd(), f"{extract_pid(files[0]) + '_' if files and extract_pid(files[0]) else ''}{out_base}.parquet"))
        ))([p.split(':',1)[1] for p in items] if ':' in items[0] else items, 
           [p.split(':',1)[0] for p in items] if ':' in items[0] else [f"cond{i+1}" for i in range(len(items))])
    ))(a[1:-1], a[-1]) if len(a) >= 3 else (print(f"Aggregate multiple condition parquets into single plot-ready output.\n[concatenating] Usage: python {a[0]} <path1> <path2> ... <out_basename> OR <label1:path1> <label2:path2> ... <out_basename>"), sys.exit(1))
)(sys.argv)
