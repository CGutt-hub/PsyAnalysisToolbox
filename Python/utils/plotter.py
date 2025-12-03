import polars as pl, matplotlib.pyplot as plt, sys, os, shutil, tempfile, re
from matplotlib.backends.backend_pdf import PdfPages

sanitize = lambda v: re.sub(r"[^A-Za-z0-9._-]", "_", str(v))
to_lst = lambda x: x.to_list() if isinstance(x, pl.Series) else (x if isinstance(x, list) else [])
attach = lambda t, o, i: ((lambda r, w: ([w.add_page(p) for p in r.pages], w.add_metadata({"/Producer": "EmotiView", "/Conformance": "/PDF/A-1b"}), w.add_attachment(os.path.basename(i), open(i, 'rb').read()), (lambda f: (w.write(f), f.close()))(open(o, 'wb')), os.remove(t), o))(__import__('pypdf').PdfReader(t), __import__('pypdf').PdfWriter()) if __import__('importlib').util.find_spec('pypdf') else (shutil.move(t, o), o)[-1])

def plot(df, pdf_path):
    """Generic plotter: handles concatenated data from concatenating_processor."""
    print(f"[PLOT] Plotting started: {pdf_path}")
    pdf = PdfPages(pdf_path)
    row = df.to_dicts()[0]
    x_data, y_data, y_var, labels, plot_type = to_lst(row['x_data']), to_lst(row['y_data']), to_lst(row.get('y_var', [])), to_lst(row.get('labels', [])), row['plot_type']
    is_concat = x_data and isinstance(x_data[0], (list, tuple))
    colors = ['dimgray', 'darkgray', 'gray', 'lightgray', 'silver']
    lbl = lambda i: labels[i] if i < len(labels) else f'Dataset {i+1}'
    
    print(f"[PLOT] Plot type: {plot_type}, Concatenated: {is_concat}, Labels: {labels if labels else 'none'}")
    
    # Grid layout for line_grid: separate subplot per condition
    if plot_type == 'line_grid' and is_concat:
        n_plots = len(x_data)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        print(f"[PLOT] Creating grid: {n_rows}x{n_cols} for {n_plots} conditions")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for i, (xd, yd, yv) in enumerate(zip(x_data, y_data, y_var if y_var else [None]*len(x_data))):
            ax = axes[i]
            xd_list, yd_list = to_lst(xd), to_lst(yd)
            ax.plot(xd_list, yd_list, linewidth=2.5, alpha=0.85, color='dimgray')
            
            # Add shaded error region if variance provided
            if yv is not None:
                yv_list = to_lst(yv)
                import numpy as np
                # Filter out None values
                if yv_list and all(v is not None for v in yv_list):
                    ax.fill_between(xd_list, [y - e for y, e in zip(yd_list, yv_list)], 
                                   [y + e for y, e in zip(yd_list, yv_list)], alpha=0.3, color='dimgray')
            
            ax.set_title(lbl(i), fontsize=14, fontweight='bold')
            ax.set_xlabel(row.get('x_label', ''), fontsize=12)
            ax.set_ylabel(row.get('y_label', ''), fontsize=12)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
            [ax.spines[s].set_visible(False) for s in ['top', 'right']]
            [ax.spines[s].set_linewidth(1.2) for s in ['left', 'bottom']]
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
    else:
        # Original single-plot layout
        fig, ax = plt.subplots(figsize=(12, 6))
        
        (([ax.plot(to_lst(xd), to_lst(yd), linewidth=2.5, label=lbl(i), alpha=0.85, color=colors[i % len(colors)]) for i, (xd, yd) in enumerate(zip(x_data, y_data))], ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='gray')) if plot_type == 'line' else
         ([ax.scatter(to_lst(xd), to_lst(yd), s=50, label=lbl(i), alpha=0.7, color=colors[i % len(colors)]) for i, (xd, yd) in enumerate(zip(x_data, y_data))], ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='gray')) if plot_type == 'scatter' else
         (lambda n_cond, cats, w: ([ax.bar([i + (j - len(cats)/2 + 0.5) * w for i in range(n_cond)], [to_lst(y_data[i])[j] for i in range(n_cond)], width=w, label=cats[j], yerr=[to_lst(y_var[i])[j] if i < len(y_var) and j < len(to_lst(y_var[i])) else 0 for i in range(n_cond)] if y_var else None, color=colors[j % len(colors)], alpha=0.85, capsize=4, error_kw={'linewidth': 1.5}) for j in range(len(cats))], ax.set_xticks(range(n_cond)), ax.set_xticklabels(labels, fontsize=11), ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')))(len(labels), to_lst(x_data[0]), 0.75 / len(to_lst(x_data[0])) if len(to_lst(x_data[0])) > 0 else 0.75) if plot_type == 'bar' else None) if is_concat else (
         ax.plot(x_data, y_data, linewidth=2.5, alpha=0.85, color='dimgray') if plot_type == 'line' else
         ax.scatter(x_data, y_data, s=50, alpha=0.7, color='dimgray') if plot_type == 'scatter' else
         (ax.bar(range(len(y_data)), y_data, yerr=y_var if y_var else None, color='dimgray', alpha=0.85, capsize=4, error_kw={'linewidth': 1.5}), ax.set_xticks(range(len(x_data))), ax.set_xticklabels(x_data, rotation=45, ha='right', fontsize=10)))
        
        ax.set_xlabel(row.get('x_axis') or row.get('x_label', ''), fontsize=13, fontweight='medium')
        ax.set_ylabel(row.get('y_axis') or row.get('y_label', ''), fontsize=13, fontweight='medium')
        (lambda yt: (ax.set_yticks(list(range(1, len(yt) + 1))), ax.set_yticklabels(yt, fontsize=9), ax.set_ylim(0.5, len(yt) + 0.5)) if yt and isinstance(yt, (list, tuple)) else ax.tick_params(labelsize=11))(row.get('y_ticks'))
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8); [ax.spines[s].set_visible(False) for s in ['top', 'right']]; [ax.spines[s].set_linewidth(1.2) for s in ['left', 'bottom']]
        plt.tight_layout()
    
    pdf.savefig(fig, bbox_inches='tight', dpi=300); plt.close(fig); pdf.close()
    print(f"[PLOT] Plotting finished: {pdf_path}")
    return pdf_path

def run(inp, out_dir, pre):
    print(f"[PLOT] Input: {inp}, Output dir: {out_dir}, Prefix: {pre}")
    df = pl.read_parquet(inp); os.makedirs(out_dir, exist_ok=True)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf'); tf_path = (plot(df, tf.name), tf.close(), tf.name)[2]
    out_pdf = os.path.join(out_dir, f"{pre}.pdf")
    comb_pq = os.path.join(os.getcwd(), f"{sanitize(pre)}_data.parquet")
    df.write_parquet(comb_pq)
    # Attach data to PDF if pypdf available, then clean up temp files
    if __import__('importlib').util.find_spec('pypdf'):
        attach(tf_path, out_pdf, comb_pq)
        os.remove(comb_pq)  # Remove temp data file after embedding in PDF
    else:
        shutil.copy2(tf_path, out_pdf)
        os.remove(tf_path)
    # Write signal file in workspace root for nextflow
    sig = f"{sanitize(pre)}_plot.parquet"
    pl.DataFrame({'signal': [1], 'source_parquet': [os.path.basename(inp)], 'output_prefix': [pre], 'pdf_path': [out_pdf]}).write_parquet(sig + '.tmp')
    os.replace(sig + '.tmp', sig)
    print(f"[PLOT] Output PDF: {out_pdf}")
    print(f"[PLOT] Signal file: {sig}")
    return out_pdf

if __name__ == '__main__': (lambda a: run(a[1], a[2], a[3] if len(a) > 3 else "plot") if len(a) >= 3 else (print("Usage: python plotter.py <input.parquet> <output_dir> [prefix]"), sys.exit(1)))(sys.argv)
