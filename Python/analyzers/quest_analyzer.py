import sys, ast, os, re, fnmatch, polars as pl, statistics

usage = lambda: print("Usage: python quest_analyzer.py <input_parquet> <cherry_pick_param> <output_prefix>") or sys.exit(1)
get_output_filename = lambda input_file, prefix: os.path.join(os.getcwd(), f"{prefix}_{os.path.splitext(os.path.basename(input_file))[0]}_quest.parquet")

def parse_param_string(s: str):
    """Parse parameter string, handling both Python literals and unquoted selector strings."""
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        # Auto-quote unquoted selector tokens like "Level: 2" or "movieFilename: NEG*"
        def _quote_token(m):
            return f"{m.group('pre')}'{m.group('tok')}'"
        pat = re.compile(r"(?P<pre>(?:\A|\[|,)\s*)(?P<tok>[A-Za-z0-9_\-]+:\s*[^,\]\}]+)")
        s_fixed = re.sub(pat, _quote_token, s)
        try:
            return ast.literal_eval(s_fixed)
        except Exception as e:
            raise RuntimeError(f'Failed to parse parameter: {e}\nOriginal: {s}\nTried: {s_fixed}')


def dotted_get(d, path):
    """Get nested dict value using dotted path notation."""
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def run(input_parquet, cherry_pick_param, output_prefix):
    """
    Cherry-pick questionnaire data from hierarchical structure and prepare for plotting.
    
    Process:
    1. Load parquet with hierarchy column
    2. Parse parameters: [selector1, selector2, ..., {x-axis, y-axis, data}]
    3. Traverse tree using selectors to filter and descend branches
    4. At target depth, collect all data from matching entries
    5. Aggregate by x-axis values (mean, stdev)
    6. Output plot-ready structure as parquet
    """
    
    # Load hierarchy
    df = pl.read_parquet(input_parquet)
    if 'hierarchy' not in df.columns:
        raise RuntimeError('Input parquet must contain "hierarchy" column')
    nested = df['hierarchy'][0]
    if not isinstance(nested, dict) or 'level' not in nested:
        raise RuntimeError('Expected hierarchy.level in parquet')
    
    root_list = nested['level']
    
    # Parse parameters (support @file for complex params)
    if isinstance(cherry_pick_param, str) and cherry_pick_param.startswith('@'):
        with open(cherry_pick_param[1:], 'r', encoding='utf-8') as fh:
            param = parse_param_string(fh.read())
    else:
        param = parse_param_string(cherry_pick_param)
    
    if not isinstance(param, list) or not param:
        raise RuntimeError('Expected parameter to be a non-empty list')
    
    # Flatten nested lists: ["sel1", ["sel2", {dict}]] → ["sel1", "sel2", {dict}]
    def flatten_param(p):
        result = []
        for el in p:
            if isinstance(el, list):
                result.extend(flatten_param(el))
            else:
                result.append(el)
        return result
    
    param = flatten_param(param)
    
    # Last element MUST be dict with plot metadata
    inner = param[-1]
    if not isinstance(inner, dict):
        raise RuntimeError('Last parameter must be dict with "x-axis" and "data" keys')
    
    x_axis_key = inner.get('x-axis') or inner.get('x_axis')
    y_axis_keys = inner.get('y-axis') or inner.get('y_axis')
    data_key = inner.get('data')
    
    if x_axis_key is None or data_key is None:
        raise RuntimeError('Parameter dict must contain "x-axis" and "data" keys')
    
    # Extract selector strings (all elements before final dict)
    selectors = [el for el in param[:-1] if isinstance(el, str) and ':' in el]
    if not selectors:
        raise RuntimeError('Expected at least one selector (e.g., "Level: 2" or "movieFilename: NEU*")')
    
    # TRAVERSE: Apply selectors sequentially to filter and descend branches
    current_list = root_list
    
    for sel_idx, selector in enumerate(selectors):
        sel_key, sel_pat = [s.strip() for s in selector.split(':', 1)]
        
        # Filter: keep only entries matching this selector
        matched = []
        for entry in current_list:
            if isinstance(entry, dict) and sel_key in entry:
                val = entry[sel_key]
                if sel_pat == '*' or fnmatch.fnmatch(str(val), sel_pat):
                    matched.append(entry)
        
        if not matched:
            raise RuntimeError(f'No matches for selector "{selector}" at level {sel_idx + 1}')
        
        # Descend: collect all _branch children from matched entries
        next_list = []
        for entry in matched:
            if '_branch' in entry and isinstance(entry['_branch'], list):
                next_list.extend(entry['_branch'])
        
        # After last selector: target depth reached
        if sel_idx == len(selectors) - 1:
            current_list = next_list if next_list else matched
            break
        
        # Continue descending for non-final selectors
        if not next_list:
            raise RuntimeError(f'No _branch found in matched entries for selector "{selector}"')
        current_list = next_list
    
    # COLLECT: Extract data from all entries at target depth
    conds = {}  # {x_value: [data_values]}
    x_order = []  # Preserve order of x-axis values
    y_ticks = None  # Y-axis tick labels
    
    for entry in current_list:
        if not isinstance(entry, dict):
            raise RuntimeError(f'Expected dict at target depth, got {type(entry).__name__}')
        
        # Get x-axis value (question text, emotion label, etc.)
        xv = dotted_get(entry, x_axis_key) if '.' in x_axis_key else entry.get(x_axis_key)
        if xv is None:
            raise RuntimeError(f'Missing x-axis key "{x_axis_key}" in entry')
        xv_s = str(xv)
        if xv_s not in x_order:
            x_order.append(xv_s)
        
        # Get y-axis tick labels (once)
        if y_ticks is None and y_axis_keys:
            ticks = []
            keys = y_axis_keys if isinstance(y_axis_keys, (list, tuple)) else [y_axis_keys]
            for k in keys:
                mv = entry.get(k)
                if mv is not None:
                    ticks.append(str(mv))
            if ticks:
                # Deduplicate while preserving order
                seen = set()
                y_ticks = [x for x in ticks if not (x in seen or seen.add(x))]
        
        # Get data value (response value)
        dv = dotted_get(entry, data_key) if '.' in data_key else entry.get(data_key)
        if dv is None:
            raise RuntimeError(f'Missing data key "{data_key}" in entry')
        if isinstance(dv, (dict, list, tuple)):
            raise RuntimeError(f'Expected scalar for data key "{data_key}", got {type(dv).__name__}')
        
        try:
            num = float(dv)
        except Exception:
            raise RuntimeError(f'Non-numeric data value: {dv!r}')
        
        # Group by x-axis value
        conds.setdefault(xv_s, []).append(num)
    
    # AGGREGATE: Calculate statistics for each x-axis value
    x_data = []
    y_data = []
    y_var = []
    counts = []
    
    for xl in x_order:
        vals = conds.get(xl, [])
        x_data.append(xl)
        if vals:
            y_data.append(statistics.mean(vals))
            y_var.append(statistics.stdev(vals) if len(vals) > 1 else None)
            counts.append(len(vals))
        else:
            y_data.append(None)
            y_var.append(None)
            counts.append(0)
    
    # Validation
    if not x_data:
        raise RuntimeError('No x-axis data found - check selectors and hierarchy structure')
    if y_axis_keys and not y_ticks:
        raise RuntimeError('y-axis keys provided but no y_ticks extracted')
    
    # OUTPUT: Create plot-ready structure
    out = {
        'x_data': x_data,
        'y_data': y_data,
        'y_var': y_var,
        'y_ticks': y_ticks,
        'x_axis': x_axis_key,
        'plot_type': 'bar',
        'counts_per_x': counts,
        'count': sum(counts),
    }
    
    out_path = get_output_filename(input_parquet, output_prefix)
    pl.DataFrame([out]).write_parquet(out_path)
    print(f'[QUEST] Wrote {out_path} | Total: {sum(counts)} values across {len(x_data)} categories')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        usage()
    try:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(f'[QUEST] ERROR: {e}')
        sys.exit(1)
