import polars as pl, matplotlib.pyplot as plt, sys, os, shutil, tempfile, traceback, re
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    usage = lambda: print("Usage: python plotter.py <input_parquet> <output_dir> [output_prefix]") or sys.exit(1)
    get_output_filename = lambda output_dir, output_prefix: os.path.join(output_dir, f"{output_prefix}.pdf")
    sanitize = lambda v: re.sub(r"[^A-Za-z0-9._-]", "_", str(v))

    def create_plot(df, temp_pdf_path):
        """Create a multi-page PDF at temp_pdf_path from rows in df.

        Accepts analyzer-style rows where each row may contain list-valued
        'x_data' and 'y_data', optional 'y_var' and 'y_ticks'. If 'y_ticks'
        is present we render horizontal bars and label the numeric axis with
        those ticks (useful for Likert-style word labels)."""
        pdf = PdfPages(temp_pdf_path)
        try:
            rows = df.to_dicts()
            for row in rows:
                plot_type = row.get('plot_type', 'bar')
                x_data = row.get('x_data') or []
                y_data = row.get('y_data') or []
                y_var = row.get('y_var') or None
                y_ticks = row.get('y_ticks') or None
                # Per user preference: no title should be drawn on the figure.
                # Keep plot content minimal; do not include the analyzer 'key' or other identifier.
                title = None

                fig, ax = plt.subplots(figsize=(12, 8))

                # normalize list-like fields
                if isinstance(x_data, pl.Series):
                    x_data = x_data.to_list()
                if isinstance(y_data, pl.Series):
                    y_data = y_data.to_list()
                if isinstance(y_var, pl.Series):
                    y_var = y_var.to_list()

                # if y_var provided and is shorter/same length as y_data, use as error
                err = None
                if y_var and isinstance(y_var, (list, tuple)):
                    err = [v if v is not None else 0 for v in y_var]

                # Bar charts
                if plot_type == 'bar':
                    # If y_ticks present -> use horizontal bars and label numeric axis with y_ticks
                    if y_ticks and isinstance(y_ticks, (list, tuple)):
                        # Vertical bars with Likert-style y-axis labels.
                        labels = [str(x)[:50] + '...' if len(str(x)) > 50 else str(x) for x in x_data]
                        vals = [0 if v is None else v for v in y_data]
                        positions = list(range(len(vals)))
                        if err:
                            ax.bar(positions, vals, yerr=err, color='dimgray', alpha=0.8)
                        else:
                            ax.bar(positions, vals, color='dimgray', alpha=0.8)
                        ax.set_xticks(positions)
                        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                        # map numeric y-axis ticks to y_ticks labels (assume 1..N scale)
                        n_ticks = len(y_ticks)
                        ax.set_yticks(list(range(1, n_ticks + 1)))
                        ax.set_yticklabels(y_ticks, rotation=0, fontsize=9)
                        ax.set_xlabel(row.get('x_axis', 'Condition'))
                        # Only set y-axis label if provided; otherwise leave blank for publication-ready plots
                        y_label = row.get('y_axis') if 'y_axis' in row else None
                        ax.set_ylabel(y_label or '')
                    else:
                        # vertical bars
                        labels = [str(x)[:50] + '...' if len(str(x)) > 50 else str(x) for x in x_data]
                        vals = [0 if v is None else v for v in y_data]
                        positions = list(range(len(vals)))
                        if err:
                            ax.bar(positions, vals, yerr=err, color='dimgray', alpha=0.8)
                        else:
                            ax.bar(positions, vals, color='dimgray', alpha=0.8)
                        ax.set_xticks(positions)
                        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                        ax.set_xlabel(row.get('x_axis', 'Condition'))
                        y_label = row.get('y_axis') if 'y_axis' in row else None
                        ax.set_ylabel(y_label or '')

                elif plot_type == 'line':
                    ax.plot(range(len(y_data)), y_data, color='black', linewidth=2, marker='o', markersize=4)
                    ax.set_xlabel(row.get('x_axis', 'Index'))
                    y_label = row.get('y_axis') if 'y_axis' in row else None
                    ax.set_ylabel(y_label or '')

                elif plot_type == 'scatter':
                    ax.scatter(range(len(y_data)), y_data, color='gray', alpha=0.6, s=50)
                    ax.set_xlabel(row.get('x_axis', 'Index'))
                    y_label = row.get('y_axis') if 'y_axis' in row else None
                    ax.set_ylabel(y_label or '')

                # Do not set any title or footer text (plots used for papers should be label-free).
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close(fig)
        finally:
            pdf.close()
        return temp_pdf_path

    attach_impl = (lambda t, out, inp: (
        (lambda reader, writer: (
            [writer.add_page(p) for p in reader.pages],
            writer.add_metadata({
                "/Producer": "EmotiView Analysis Pipeline",
                "/Title": f'Analysis Plot from {os.path.basename(inp)}',
                "/Subject": f'Generated from parquet data: {inp}',
                "/Creator": "EmotiView Analysis Pipeline",
                "/Conformance": "/PDF/A-1b"
            }),
            writer.add_attachment(os.path.basename(inp), open(inp, 'rb').read()),
            (lambda f: (writer.write(f), f.close()))(open(out, 'wb')),
            (os.remove(t) if os.path.exists(t) else None),
            out
        ))(__import__('pypdf').PdfReader(t), __import__('pypdf').PdfWriter())
    )) if __import__('importlib').util.find_spec('pypdf') else (lambda t, out, inp: (shutil.move(t, out), out))

    def run(input_parquet, output_dir, output_prefix):
        print(f"[UTIL] Plotting started for file: {input_parquet}")
        # Read the parquet once
        df = pl.read_parquet(input_parquet)
        # Default behavior: if plot_type is missing, assume 'bar'
        if 'plot_type' not in df.columns:
            df = df.with_columns(pl.Series('plot_type', ['bar'] * df.height)) if df.height > 0 else df

        os.makedirs(output_dir, exist_ok=True)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            create_plot(df, tf.name)
            tf.close()

            # Ensure we place a copy in the current working directory so Nextflow can capture it
            local_pdf = os.path.join(os.getcwd(), f"{sanitize(output_prefix)}.pdf")
            try:
                # Attach metadata and write local copy
                attach_impl(tf.name, local_pdf, input_parquet)
            except Exception:
                # fallback: plain copy
                shutil.copy2(tf.name, local_pdf)

            # Also copy into the user-requested output_dir (best-effort)
            out_pdf = get_output_filename(output_dir, output_prefix)
            try:
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(local_pdf, out_pdf)
            except Exception:
                # do not fail the process for optional copy failures; log instead
                print(f"[UTIL] WARNING: Could not copy PDF to output_dir '{output_dir}'")

            # write sentinel parquet in the working dir (this is how the pipeline detects plot outputs)
            sig_base = os.path.splitext(os.path.basename(input_parquet))[0] + '_plot.parquet'
            pl.DataFrame({
                'signal': [1],
                'source_parquet': [os.path.basename(input_parquet)],
                'output_prefix': [output_prefix],
                'pdf_path': [out_pdf]
            }).write_parquet(sig_base + '.tmp')
            os.replace(sig_base + '.tmp', sig_base)
            return out_pdf
        finally:
            if os.path.exists(tf.name):
                try:
                    os.remove(tf.name)
                except Exception:
                    pass
    

    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            input_parquet = args[1]
            output_dir = args[2]
            output_prefix = args[3] if len(args) > 3 else "plotter"
            run(input_parquet, output_dir, output_prefix)
    except Exception as e:
        print(f"[UTIL] ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)