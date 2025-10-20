import polars as pl, matplotlib.pyplot as plt, sys, os, shutil, tempfile, traceback
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    usage = lambda: print("Usage: python plotter.py <input_parquet> <output_dir> [output_prefix]") or sys.exit(1)
    get_output_filename = lambda output_dir, output_prefix: os.path.join(output_dir, f"{output_prefix}.pdf")

    create_plot = lambda df, plot_type, temp_pdf_path: (
        (lambda fig_ax: (
            (lambda fig, ax: (
                (lambda _: (
                    (lambda y_data: (
                        (lambda bars: (
                            ax.set_xticks(range(len(bars))),
                            ax.set_xticklabels([str(x)[:50] + "..." if len(str(x)) > 50 else str(x) for x in (df["x_data"].to_list() if "x_data" in df.columns else range(len(y_data)))], rotation=45, ha='right', fontsize=8),
                            ax.set_xlabel(df["x_axis"][0] if "x_axis" in df.columns else "Items"),
                            ax.set_ylabel(df["y_axis"][0] if "y_axis" in df.columns else "Values")
                        ))(ax.bar(range(len(y_data)), y_data, color='dimgray', alpha=0.7)) if plot_type == "bar" and "x_data" in df.columns and "y_data" in df.columns else
                        (lambda value_counts: (
                            ax.bar([str(val) for val in value_counts["value"].to_list()], value_counts["counts"].to_list(), color='gray', alpha=0.7),
                            ax.set_xlabel("Values"),
                            ax.set_ylabel("Count")
                        ))(pl.Series(y_data).value_counts()) if plot_type == "bar" else
                        (lambda _: (
                            ax.plot(range(len(y_data)), y_data, color='black', linewidth=2, marker='o', markersize=4),
                            ax.set_xlabel("Index"),
                            ax.set_ylabel("Values")
                        ))(None) if plot_type == "line" else
                        (lambda _: (
                            ax.scatter(range(len(y_data)), y_data, color='gray', alpha=0.6, s=50),
                            ax.set_xlabel("Index"),
                            ax.set_ylabel("Values")
                        ))(None) if plot_type == "scatter" else None
                    ))(df["y_data"].to_list() if "y_data" in df.columns else [1] * df.height),
                    ax.grid(True, alpha=0.3),
                    ax.spines['top'].set_visible(False),
                    ax.spines['right'].set_visible(False),
                    plt.tight_layout(),
                    (lambda _: (
                        (lambda pdf: (
                            pdf.savefig(fig, bbox_inches='tight', dpi=300),
                            pdf.close(),
                            temp_pdf_path
                        ))(PdfPages(temp_pdf_path))
                    ))(None),
                    plt.close(fig),
                    temp_pdf_path
                ))(None)
            ))(fig_ax[0], fig_ax[1])
        ))(plt.subplots(figsize=(12, 8)))
    )

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

    run = lambda input_parquet, output_dir, output_prefix: (
        print(f"[UTIL] Plotting started for file: {input_parquet}") or
        (lambda df: (
            (lambda plot_type: (
                os.makedirs(output_dir, exist_ok=True),
                (lambda tf: (
                    create_plot(df, plot_type, tf.name),
                    tf.close(),
                    attach_impl(tf.name, get_output_filename(output_dir, output_prefix), input_parquet),
                    (lambda sig: (
                        pl.DataFrame({
                            'signal': [1],
                            'source_parquet': [os.path.basename(input_parquet)],
                            'output_prefix': [output_prefix]
                        }).write_parquet(sig),
                        sig
                    ))(os.path.splitext(os.path.basename(input_parquet))[0] + '_plot.parquet'),
                    get_output_filename(output_dir, output_prefix)
                ))(tempfile.NamedTemporaryFile(delete=False, suffix='.pdf'))
            ))(df["plot_type"][0] if "plot_type" in df.columns else "bar")
        ))(pl.read_parquet(input_parquet)) if "plot_type" in (lambda df: df)(pl.read_parquet(input_parquet)).columns else (
            print(f"[UTIL] ERROR: Input parquet file must contain 'plot_type' column") or
            sys.exit(1)
        )
    )

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