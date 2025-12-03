import polars as pl, sys, os, fnmatch
from typing import Optional

def analyze_scr(ip: str, condition_selector: str = "*", output_label: Optional[str] = None) -> str:
    """
    Analyze SCR data from epoched EDA, cherry-picking specific condition(s).
    
    Args:
        ip: Input parquet file with epoched EDA data
        condition_selector: Either a glob pattern (e.g., "NEG*", "Positive") or exact condition label
        output_label: Optional custom label for output filename (defaults to matched condition name)
    """
    df = pl.read_parquet(ip)
    event_col = 'event' if 'event' in df.columns else 'condition'
    eda_col = [c for c in df.columns if c not in ['time', 'sfreq', 'event', 'epoch_id', 'condition', event_col]][0]
    
    # Filter to matching condition(s)
    all_conditions = df[event_col].unique().to_list()
    
    # Try exact match first, then glob pattern
    if condition_selector in all_conditions:
        matched = [condition_selector]
    else:
        matched = [c for c in all_conditions if fnmatch.fnmatch(str(c).upper(), condition_selector.upper())]
    
    if not matched:
        raise ValueError(f"No conditions matched '{condition_selector}' from available: {all_conditions}")
    
    df = df.filter(pl.col(event_col).is_in(matched))
    print(f"[ANALYZE] Matched conditions: {matched} (selector: {condition_selector})")
    
    # Compute epoch-averaged waveforms with relative time
    df = df.with_columns([
        (pl.col('time') - pl.col('time').min().over('epoch_id')).alias('relative_time')
    ])
    
    # Average across epochs, then downsample (every 50th point for manageable file size)
    result = df.group_by([event_col, 'relative_time']).agg([
        pl.col(eda_col).mean().alias('mean_eda')
    ]).sort([event_col, 'relative_time']).with_row_index().filter(
        pl.col('index') % 50 == 0
    ).drop('index')
    
    # Group by condition and collect time series as lists (one row per condition)
    output = result.group_by(event_col).agg([
        pl.col('relative_time').alias('x_data'),
        pl.col('mean_eda').alias('y_data')
    ]).sort(event_col).with_columns([
        pl.col(event_col).cast(pl.Utf8).alias('condition'),
        pl.lit('line').alias('plot_type'),
        pl.lit('Time from onset (s)').alias('x_label'),
        pl.lit('Mean EDA (μS)').alias('y_label')
    ])
    
    # Single row output (one condition per file, like quest_analyzer)
    if len(output) != 1:
        raise ValueError(f"Expected 1 condition after filtering, got {len(output)}: {output['condition'].to_list()}")
    
    # Use custom label or condition name in output filename
    if output_label:
        cond_name = output_label.replace('/', '_').replace('\\', '_').replace(' ', '_')
    else:
        cond_name = output['condition'][0].replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    out = f"{os.path.splitext(os.path.basename(ip))[0]}_{cond_name}_scr.parquet"
    output.write_parquet(out)
    print(f"[ANALYZE] SCR output: {out} | {len(result)} points (downsampled)")
    return out

if __name__ == '__main__':
    (lambda a: [print(f"[ANALYZE] Analyzing SCR: {a[1]} (selector: {a[2] if len(a) >= 3 else '*'})"), analyze_scr(a[1], a[2] if len(a) >= 3 else "*", a[3] if len(a) >= 4 else None)] if len(a) >= 2 else [print("[ANALYZE] Usage: python scr_analyzer.py <input.parquet> [condition_selector] [output_label]"), sys.exit(1)])(sys.argv)
