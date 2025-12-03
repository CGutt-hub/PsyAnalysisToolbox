import sys, ast, os, re, fnmatch, polars as pl, statistics

parse_param = lambda s: s if not isinstance(s, str) else (lambda: ast.literal_eval(s) if not s.startswith('@') else ast.literal_eval(open(s[1:], 'r', encoding='utf-8').read()))() if s.startswith('@') or s.startswith('[') or s.startswith('{') else ast.literal_eval(re.sub(r"(?P<pre>(?:\A|\[|,)\s*)(?P<tok>[A-Za-z0-9_\-]+:\s*[^,\]\}]+)", lambda m: f"{m.group('pre')}'{m.group('tok')}'", s))
parse_entry = lambda e: tuple(x.strip() for x in e.split(':', 1)) if ':' in e else (e, None)
def get_prop_deep(n, k, depth=3):
    v = next((v for c in n.get('children', []) if c.get('entry') and (kv := parse_entry(c['entry'])) and kv[0] == k for v in [kv[1]]), None)
    if v is not None or depth <= 0: return v
    for c in n.get('children', []):
        if not c.get('entry') and c.get('children'):
            v = get_prop_deep(c, k, depth-1)
            if v is not None: return v
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
    sels = [el for el in param[:-1] if isinstance(el, str) and ':' in el]
    if not sels: raise RuntimeError('Need at least one selector')
    # Collect all structural nodes recursively from root
    def collect_structural(node):
        result = []
        for c in node.get('children', []):
            if not c.get('entry') and c.get('children'):
                result.append(c)
                result.extend(collect_structural(c))
        return result
    nodes = collect_structural(root)
    for i, sel in enumerate(sels):
        k, p = [s.strip() for s in sel.split(':', 1)]; matched = get_branches(nodes, k, p)
        print(f"[ANALYZER] Selector '{sel}' matched {len(matched)} branches")
        nodes = matched
        if i < len(sels) - 1:
            nodes = [c for n in nodes for c in n.get('children', []) if not c.get('entry') and c.get('children')]
    conds, x_ord, y_ticks = {}, [], None
    for n in nodes:
        xv = str(xv_raw) if (xv_raw := get_prop(n, x_key)) is not None else None
        if xv and xv not in x_ord: x_ord.append(xv)
        if y_ticks is None and y_keys: y_ticks = list(dict.fromkeys([str(v) for k in (y_keys if isinstance(y_keys, (list, tuple)) else [y_keys]) if (v := get_prop(n, k)) is not None]))
        if xv and (dv := get_prop(n, d_key)) is not None: conds.setdefault(xv, []).append(float(dv))
    x_data, y_data, y_var, counts = [], [], [], []
    for xl in x_ord: vals = conds.get(xl, []); x_data.append(xl); y_data.append(statistics.mean(vals) if vals else None); y_var.append(statistics.stdev(vals) if len(vals) > 1 else None); counts.append(len(vals))
    # Apply custom tick labels if provided (mapping from original to custom)
    if x_tick_labels and isinstance(x_tick_labels, dict):
        x_data = [x_tick_labels.get(x, x) for x in x_data]
    out = {'x_data': x_data, 'y_data': y_data, 'y_var': y_var, 'y_ticks': y_ticks, 'x_axis': x_label, 'plot_type': 'bar', 'counts_per_x': counts, 'count': sum(counts)}
    out_path = os.path.join(os.getcwd(), f"{out_prefix}_{os.path.splitext(os.path.basename(ip))[0]}_quest.parquet"); pl.DataFrame([out]).write_parquet(out_path); print(f"[ANALYZER] Output: {out_path} | {sum(counts)} values, {len(x_data)} categories")

if __name__ == '__main__': (lambda a: analyze(a[1], a[2], a[3]) if len(a) >= 4 else (print("Usage: python quest_analyzer.py <input_parquet> <cherry_pick_param> <output_prefix>"), sys.exit(1)))(sys.argv)
