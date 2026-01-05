import sys, ast, os, re, fnmatch, polars as pl, statistics

parse_param = lambda s: s if not isinstance(s, str) else (lambda: ast.literal_eval(s) if not s.startswith('@') else ast.literal_eval(open(s[1:], 'r', encoding='utf-8').read()))() if s.startswith('@') or s.startswith('[') or s.startswith('{') else ast.literal_eval(re.sub(r"(?P<pre>(?:\A|\[|,)\s*)(?P<tok>[A-Za-z0-9_\-]+:\s*[^,\]\}]+)", lambda m: f"{m.group('pre')}'{m.group('tok')}'", s))
parse_entry = lambda e: tuple(x.strip() for x in e.split(':', 1)) if ':' in e else (e, None)
def get_prop_deep(n, k):
    """Search for property k in node n - checks direct children (entries) first, then recursively searches structural descendants."""
    # Check direct entry children first - use 'value' field directly, not parsed from entry string
    for c in n.get('children', []):
        if c.get('entry') == k:
            return c.get('value')
    # Then recursively check structural children
    for c in n.get('children', []):
        if not c.get('entry') and c.get('children'):
            v = get_prop_deep(c, k)
            if v is not None:
                return v
    return None
get_prop = get_prop_deep
get_branches = lambda nodes, k, pat=None: [b for b in [n for n in nodes if not n.get('entry') and n.get('children')] if (v := get_prop(b, k)) is not None and (pat is None or fnmatch.fnmatch(str(v), pat))]
flatten = lambda p: [x for el in p for x in (flatten(el) if isinstance(el, list) else [el])]

def analyze(ip, param_str, out_prefix):
    print(f"[ANALYZER] Processing: {ip}"); root = pl.read_parquet(ip)['data'][0]; param = flatten(parse_param(param_str))
    if not isinstance(param[-1], dict): raise RuntimeError('Last param must be dict')
    inner = param[-1]; x_key, y_keys, d_key = inner.get('x-axis') or inner.get('x_axis'), inner.get('y-axis') or inner.get('y_axis'), inner.get('data')
    x_label = inner.get('x-axis-label') or inner.get('x_axis_label') or x_key  # Optional custom x-axis label
    x_tick_labels = inner.get('x-tick-labels') or inner.get('x_tick_labels')  # Optional custom tick labels (dict mapping)
    if not x_key or not d_key: raise RuntimeError('Need x-axis and data keys')
    
    # Extract condition patterns (non-dict, non-selector strings before the dict)
    cond_patterns = [el for el in param[:-1] if isinstance(el, str) and ':' not in el]
    sels = [el for el in param[:-1] if isinstance(el, str) and ':' in el]
    
    # Collect all structural nodes recursively from root
    def collect_structural(node):
        result = []
        for c in node.get('children', []):
            if not c.get('entry') and c.get('children'):
                result.append(c)
                result.extend(collect_structural(c))
        return result
    nodes = collect_structural(root)
    
    # Apply selectors if provided
    if sels:
        for i, sel in enumerate(sels):
            k, p = [s.strip() for s in sel.split(':', 1)]; matched = get_branches(nodes, k, p)
            print(f"[ANALYZER] Selector '{sel}' matched {len(matched)} branches")
            nodes = matched
            if i < len(sels) - 1:
                nodes = [c for n in nodes for c in n.get('children', []) if not c.get('entry') and c.get('children')]
    
    # If condition patterns provided, process each condition separately (creates separate output files)
    if cond_patterns:
        print(f"[ANALYZER] Condition patterns: {cond_patterns}")
        base = os.path.splitext(os.path.basename(ip))[0]
        folder_name = f"{base}_{out_prefix}"
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        condition_count = 0
        for idx, pattern in enumerate(cond_patterns, 1):
            cond_name = pattern.replace('*', '').upper()
            # Find nodes with entries matching the pattern, then get their nephews (structural children)
            parent_nodes = [n for n in nodes if any(c.get('entry') and fnmatch.fnmatch(str(c.get('value', '')), pattern) for c in n.get('children', []) if c.get('entry'))]
            # Get nephews (structural children of matched parents)
            nephew_nodes = [child for parent in parent_nodes for child in parent.get('children', []) if not child.get('entry') and child.get('children')]
            print(f"[ANALYZER]   Pattern '{pattern}' -> '{cond_name}': {len(parent_nodes)} parents, {len(nephew_nodes)} nephews")
            
            if not nephew_nodes:
                print(f"[ANALYZER]   WARNING: No nephews found for pattern '{pattern}'")
                continue
            
            # Process nephews for this condition
            conds, x_ord, y_ticks = {}, [], None
            for n in nephew_nodes:
                xv = str(xv_raw) if (xv_raw := get_prop(n, x_key)) is not None else None
                if xv and xv not in x_ord: x_ord.append(xv)
                if y_ticks is None and y_keys: y_ticks = list(dict.fromkeys([str(v) for k in (y_keys if isinstance(y_keys, (list, tuple)) else [y_keys]) if (v := get_prop(n, k)) is not None]))
                if xv and (dv := get_prop(n, d_key)) is not None: conds.setdefault(xv, []).append(float(dv))
            
            if not conds:
                print(f"[ANALYZER]   WARNING: No data extracted for condition '{cond_name}'")
                continue
            
            x_data, y_data, y_var, counts = [], [], [], []
            for xl in x_ord: vals = conds.get(xl, []); x_data.append(xl); y_data.append(statistics.mean(vals) if vals else None); y_var.append(statistics.stdev(vals) if len(vals) > 1 else None); counts.append(len(vals))
            if x_tick_labels and isinstance(x_tick_labels, dict):
                x_data = [x_tick_labels.get(x, x) for x in x_data]
            
            out = {'x_data': x_data, 'y_data': y_data, 'y_var': y_var, 'y_ticks': y_ticks, 'x_axis': x_label, 'plot_type': 'grid', 'counts_per_x': counts, 'count': sum(counts), 'condition': cond_name}
            out_path = os.path.join(folder_path, f"{base}_{out_prefix}{idx}.parquet")
            pl.DataFrame([out]).write_parquet(out_path)
            print(f"[ANALYZER]   Output: {out_path} | {sum(counts)} values, {len(x_data)} categories | condition: {cond_name}")
            condition_count += 1
        
        # Write signalling file in workspace root
        signal_path = os.path.join(os.getcwd(), f"{folder_name}.parquet")
        pl.DataFrame([{'signal': 1, 'source': os.path.basename(ip), 'conditions': condition_count, 'folder_path': os.path.abspath(folder_path)}]).write_parquet(signal_path)
        print(f"[ANALYZER] Signal file: {signal_path} | {condition_count} conditions")
        print(signal_path)
        return signal_path
    else:
        # No condition patterns - search for x-axis entries and get sibling data values
        print(f"[ANALYZER] No condition patterns - searching for x-axis entries as siblings")
        
        # Recursively find all nodes in the tree
        def get_all_nodes(nodes):
            result = []
            for n in nodes:
                result.append(n)
                result.extend(get_all_nodes(n.get('children', [])))
            return result
        
        all_nodes = get_all_nodes(nodes)
        print(f"[ANALYZER] Total nodes in tree: {len(all_nodes)}")
        
        # Find parent nodes that contain x-axis entries
        parent_nodes = []
        for n in all_nodes:
            children = n.get('children', [])
            # Check if any child has an entry matching x_key or y_keys
            has_x_entry = any(c.get('entry') == x_key for c in children)
            has_y_entry = y_keys and any(c.get('entry') in (y_keys if isinstance(y_keys, (list, tuple)) else [y_keys]) for c in children)
            if has_x_entry or has_y_entry:
                parent_nodes.append(n)
        
        print(f"[ANALYZER] Found {len(parent_nodes)} parent nodes with x-axis or y-axis entries")
        
        conds, x_ord, y_ticks = {}, [], None
        for parent in parent_nodes:
            children = parent.get('children', [])
            # Get x-axis value from entry
            xv = None
            for c in children:
                if c.get('entry') == x_key:
                    xv = str(c.get('value'))
                    if xv and xv not in x_ord:
                        x_ord.append(xv)
                    break
            
            # Get y_ticks from y_keys entries
            if y_ticks is None and y_keys:
                y_list = y_keys if isinstance(y_keys, (list, tuple)) else [y_keys]
                y_ticks = []
                for c in children:
                    if c.get('entry') in y_list:
                        val = str(c.get('value'))
                        if val and val not in y_ticks:
                            y_ticks.append(val)
            
            # Get data value from sibling entry
            for c in children:
                if c.get('entry') == d_key:
                    dv = c.get('value')
                    if xv and dv is not None:
                        conds.setdefault(xv, []).append(float(dv))
                    break
        
        print(f"[ANALYZER] Extracted data: {len(x_ord)} x-values, {sum(len(v) for v in conds.values())} total data points")
        
        x_data, y_data, y_var, counts = [], [], [], []
        for xl in x_ord:
            vals = conds.get(xl, [])
            x_data.append(xl)
            y_data.append(statistics.mean(vals) if vals else None)
            y_var.append(statistics.stdev(vals) if len(vals) > 1 else None)
            counts.append(len(vals))
        
        if x_tick_labels and isinstance(x_tick_labels, dict):
            x_data = [x_tick_labels.get(x, x) for x in x_data]
        
        out = {'x_data': x_data, 'y_data': y_data, 'y_var': y_var, 'y_ticks': y_ticks, 'x_axis': x_label, 'plot_type': 'bar', 'counts_per_x': counts, 'count': sum(counts)}
        base = os.path.splitext(os.path.basename(ip))[0]
        out_path = os.path.join(os.getcwd(), f"{base}_{out_prefix}.parquet")
        pl.DataFrame([out]).write_parquet(out_path)
        print(f"[ANALYZER] Output: {out_path} | {sum(counts)} values, {len(x_data)} categories")
        print(out_path)
        return out_path

if __name__ == '__main__': (lambda a: analyze(a[1], a[2], a[3]) if len(a) >= 4 else (print("Usage: python quest_analyzer.py <input_parquet> <cherry_pick_param> <output_prefix>"), sys.exit(1)))(sys.argv)
