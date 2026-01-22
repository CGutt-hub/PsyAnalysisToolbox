import polars as pl, matplotlib.pyplot as plt, sys, os, shutil, tempfile, re
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sanitize = lambda v: re.sub(r"[^A-Za-z0-9._-]", "_", str(v))
to_lst = lambda x: x.to_list() if isinstance(x, pl.Series) else (x if isinstance(x, list) else [])
truncate = lambda s, max_len=40: (s if len(s) <= max_len else s[:max_len-3] + '...') if isinstance(s, str) else s
# Convert None values to NaN for matplotlib compatibility
safe_yerr = lambda yerr: [np.nan if x is None else x for x in yerr] if yerr else None
attach = lambda t, o, i: ((lambda r, w: ([w.add_page(p) for p in r.pages], w.add_metadata({"/Producer": "EmotiView", "/Conformance": "/PDF/A-1b"}), w.add_attachment(os.path.basename(i), open(i, 'rb').read()), (lambda f: (w.write(f), f.close()))(open(o, 'wb')), os.remove(t), o))(__import__('pypdf').PdfReader(t), __import__('pypdf').PdfWriter()) if __import__('importlib').util.find_spec('pypdf') else (shutil.move(t, o), o)[-1])

def plot(df, pdf_path):
    """Generic plotter: handles concatenated data from concatenating_processor."""
    print(f"[plotter] Plotting started: {pdf_path}")
    pdf = PdfPages(pdf_path)
    row = df.to_dicts()[0]
    x_data, y_data, y_var, labels = to_lst(row['x_data']), to_lst(row['y_data']), to_lst(row.get('y_var', [])), to_lst(row.get('labels', []))
    # Extract plot_type - handle nested lists from concatenation
    raw_plot_type = row['plot_type']
    if isinstance(raw_plot_type, list):
        plot_type = raw_plot_type[0] if raw_plot_type else 'bar'
        if isinstance(plot_type, list):
            plot_type = plot_type[0] if plot_type else 'bar'
    else:
        plot_type = raw_plot_type
    is_concat = x_data and isinstance(x_data[0], (list, tuple))
    colors = ['dimgray', 'darkgray', 'gray', 'lightgray', 'silver']
    lbl = lambda i: labels[i] if i < len(labels) else f'Dataset {i+1}'
    
    print(f"[plotter] Plot type: {plot_type}, Concatenated: {is_concat}, Labels: {labels if labels else 'none'}")
    
    # Grid layout for line_grid or grid: separate subplot per condition
    if (plot_type == 'line_grid' or plot_type == 'grid') and is_concat:
        n_plots = len(x_data)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        print(f"[plotter] Creating grid: {n_rows}x{n_cols} for {n_plots} conditions")
        # Add extra space at top for suptitle
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows + 0.5))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Calculate global axis limits for consistent scaling across all subplots
        all_x_data, all_y_data = [], []
        for xd, yd in zip(x_data, y_data):
            xd_list, yd_list = to_lst(xd), to_lst(yd)
            all_x_data.extend(xd_list)
            all_y_data.extend(yd_list)
        
        # Calculate global limits for both x and y axes
        y_min, y_max = min(all_y_data), max(all_y_data)
        y_range = y_max - y_min
        y_margin = y_range * 0.1  # 10% margin
        global_y_lim = (y_min - y_margin, y_max + y_margin)
        
        if plot_type == 'line_grid':
            # For line plots, also calculate x-axis limits
            x_min, x_max = min(all_x_data), max(all_x_data)
            x_range = x_max - x_min
            x_margin = x_range * 0.05  # 5% margin
            global_x_lim = (x_min - x_margin, x_max + x_margin)
        else:
            # For bar plots, x-axis is categorical (no global x limit needed)
            global_x_lim = None
        
        for i, (xd, yd, yv) in enumerate(zip(x_data, y_data, y_var if y_var else [None]*len(x_data))):
            ax = axes[i]
            xd_list, yd_list = to_lst(xd), to_lst(yd)
            
            # For 'grid' type (bar plots), create bar chart
            if plot_type == 'grid':
                yerr_safe = safe_yerr(to_lst(yv)) if yv else None
                ax.bar(range(len(yd_list)), yd_list, yerr=yerr_safe, color='dimgray', alpha=0.85, capsize=4, error_kw={'linewidth': 1.5})
                ax.set_xticks(range(len(xd_list)))
                ax.set_xticklabels([truncate(x) for x in xd_list], rotation=45, ha='right', fontsize=9)
                # Move x-axis labels to bottom (important for plots with negative values like fnirs_rel)
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
                ax.spines['bottom'].set_position(('axes', 0))  # Keep spine at bottom
                # Check if y_ticks or y_labels override is set (for fixed Y-axis limits)
                yt = row.get('y_ticks')
                yl = row.get('y_labels')
                if yt and isinstance(yt, (int, float)):
                    ax.set_ylim(0.5, yt + 0.5)
                elif yl and isinstance(yl, (list, tuple)) and len(yl) > 2:
                    ax.set_ylim(0.5, len(yl) + 0.5)
                else:
                    ax.set_ylim(global_y_lim)
            else:
                # For 'line_grid' type, create line plot
                ax.plot(xd_list, yd_list, linewidth=2.5, alpha=0.85, color='dimgray')
                
                # Add shaded error region if variance provided
                if yv is not None:
                    yv_list = to_lst(yv)
                    import numpy as np
                    # Filter out None values
                    if yv_list and all(v is not None for v in yv_list):
                        ax.fill_between(xd_list, [y - e for y, e in zip(yd_list, yv_list)], 
                                       [y + e for y, e in zip(yd_list, yv_list)], alpha=0.3, color='dimgray')
                
                if global_x_lim:
                    ax.set_xlim(global_x_lim)
                # Check if y_ticks override is set (for fixed Y-axis limits across participants)
                yt = row.get('y_ticks')
                if yt and isinstance(yt, (int, float)):
                    ax.set_ylim(0, yt)
                else:
                    ax.set_ylim(global_y_lim)
            
            ax.set_title(lbl(i), fontsize=14, fontweight='bold')
            ax.set_xlabel(row.get('x_label', '') if plot_type == 'line_grid' else row.get('x_axis', ''), fontsize=12)
            ax.set_ylabel(row.get('y_label', ''), fontsize=12)
            # Apply y_ticks/y_labels for questionnaire scale labels (grid/bar plots only)
            yt = row.get('y_ticks')
            yl = row.get('y_labels')
            if plot_type == 'grid':
                if yt and isinstance(yt, int):
                    # y_ticks is int: use as scale max
                    ax.set_yticks(list(range(1, yt + 1)))
                    if yl and isinstance(yl, (list, tuple)) and len(yl) == 2:
                        # 2 labels: endpoints only (renamed to avoid shadowing condition labels)
                        ytick_labels = [''] * yt
                        ytick_labels[0] = str(yl[0])
                        ytick_labels[-1] = str(yl[1])
                        ax.set_yticklabels(ytick_labels, fontsize=9)
                    elif yl and isinstance(yl, (list, tuple)) and len(yl) == 3:
                        # 3 labels: bottom, middle, top (requires odd y-max for true center)
                        ytick_labels = [''] * yt
                        ytick_labels[0] = str(yl[0])
                        ytick_labels[(yt - 1) // 2] = str(yl[1])
                        ytick_labels[-1] = str(yl[2])
                        ax.set_yticklabels(ytick_labels, fontsize=9)
                    # else: numeric ticks (SAM case)
                elif yl and isinstance(yl, (list, tuple)) and len(yl) > 2:
                    # Full labels list (PANAS, BISBAS): use length as scale, all labeled
                    ax.set_yticks(list(range(1, len(yl) + 1)))
                    ax.set_yticklabels(yl, fontsize=9)
                    ax.set_ylim(0.5, len(yl) + 0.5)
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
         (lambda n_cond, cats, w: ([ax.bar([i + (j - len(cats)/2 + 0.5) * w for i in range(n_cond)], [to_lst(y_data[i])[j] for i in range(n_cond)], width=w, label=cats[j], yerr=safe_yerr([to_lst(y_var[i])[j] if i < len(y_var) and j < len(to_lst(y_var[i])) else 0 for i in range(n_cond)]) if y_var else None, color=colors[j % len(colors)], alpha=0.85, capsize=4, error_kw={'linewidth': 1.5}) for j in range(len(cats))], ax.set_xticks(range(n_cond)), ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11), ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')))(len(labels), to_lst(x_data[0]), 0.75 / len(to_lst(x_data[0])) if len(to_lst(x_data[0])) > 0 else 0.75) if plot_type == 'bar' else None) if is_concat else (
         ax.plot(x_data, y_data, linewidth=2.5, alpha=0.85, color='dimgray') if plot_type == 'line' else
         ax.scatter(x_data, y_data, s=50, alpha=0.7, color='dimgray') if plot_type == 'scatter' else
         (ax.bar(range(len(y_data)), y_data, yerr=safe_yerr(y_var) if y_var else None, color='dimgray', alpha=0.85, capsize=4, error_kw={'linewidth': 1.5}), ax.set_xticks(range(len(x_data))), ax.set_xticklabels([truncate(x) for x in x_data], rotation=45, ha='right', fontsize=10)))
        
        ax.set_xlabel(row.get('x_axis') or row.get('x_label', ''), fontsize=13, fontweight='medium')
        ax.set_ylabel(row.get('y_axis') or row.get('y_label', ''), fontsize=13, fontweight='medium')
        (lambda yt: (ax.set_yticks(list(range(1, len(yt) + 1))), ax.set_yticklabels(yt, fontsize=9), ax.set_ylim(0.5, len(yt) + 0.5)) if yt and isinstance(yt, (list, tuple)) else ax.tick_params(labelsize=11))(row.get('y_ticks'))
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8); [ax.spines[s].set_visible(False) for s in ['top', 'right']]; [ax.spines[s].set_linewidth(1.2) for s in ['left', 'bottom']]
        plt.tight_layout()
    
    pdf.savefig(fig, bbox_inches='tight', dpi=300); plt.close(fig); pdf.close()
    print(f"[plotter] Plotting finished: {pdf_path}")
    return pdf_path

def run(inp, out_dir, pre):
    print(f"[plotter] Input: {inp}, Output dir: {out_dir}, Prefix: {pre}")
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
    print(f"[plotter] Output PDF: {out_pdf}")
    print(f"[plotter] Signal file: {sig}")
    print(sig)
    return out_pdf

if __name__ == '__main__': (lambda a: run(a[1], a[2], a[3] if len(a) > 3 else "plot") if len(a) >= 3 else (print("Usage: python plotter.py <input.parquet> <output_dir> [prefix]"), sys.exit(1)))(sys.argv)
