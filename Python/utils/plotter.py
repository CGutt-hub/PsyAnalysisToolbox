import polars as pl, matplotlib.pyplot as plt, sys, os
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfWriter
if __name__ == "__main__":
    usage = lambda: print("Usage: python plotter.py <input_parquet> <output_dir> [output_prefix]") or sys.exit(1)
    get_output_filename = lambda output_dir, output_prefix: os.path.join(output_dir, f"{output_prefix}.pdf")
    create_plot = lambda df, plot_type, output_path, input_parquet: (
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
                            ax.bar([str(val) for val in value_counts[""].to_list()], value_counts["count"].to_list(), color='gray', alpha=0.7),
                            ax.set_xlabel("Values"),
                            ax.set_ylabel("Count")
                        ))(pl.Series(y_data).value_counts().sort(pl.col("").cast(pl.String))) if plot_type == "bar" else
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
                    (lambda pdf: (
                        pdf.savefig(fig, bbox_inches='tight', dpi=300),
                        pdf.close(),
                        (lambda writer: (
                            writer.add_metadata({
                                "/Producer": "EmotiView Analysis Pipeline",
                                "/Title": f'Analysis Plot from {os.path.basename(input_parquet)}',
                                "/Subject": f'Generated from parquet data: {input_parquet}',
                                "/Creator": "EmotiView Analysis Pipeline",
                                "/Conformance": "/PDF/A-1b"
                            }),
                            writer.add_attachment(input_parquet, os.path.basename(input_parquet)),
                            writer.write(output_path),
                            output_path
                        ))(PdfWriter(output_path)) if PdfWriter else output_path
                    ))(PdfPages(output_path, metadata={'Title': f'Analysis Plot from {os.path.basename(input_parquet)}', 'Subject': f'Generated from parquet data: {input_parquet}', 'Creator': 'EmotiView Analysis Pipeline'})) if PdfWriter else (
                        (lambda pdf: pdf.savefig(fig, bbox_inches='tight', dpi=300, metadata={'Title': f'Analysis Plot from {os.path.basename(input_parquet)}', 'Subject': f'Generated from parquet data: {input_parquet}', 'Creator': 'EmotiView Analysis Pipeline'}))(PdfPages(output_path)),
                        output_path
                    ),
                    plt.close(fig),
                    output_path
                ))(None)
            ))(fig_ax[0], fig_ax[1])
        ))(plt.subplots(figsize=(12, 8)))
    )
    run = lambda input_parquet, output_dir, output_prefix: (
        print(f"[UTIL] Plotting started for file: {input_parquet}") or
        (lambda df: (
            (lambda plot_type: (
                os.makedirs(output_dir, exist_ok=True),
                (lambda output_path: (
                    create_plot(df, plot_type, output_path, input_parquet),
                    (lambda conditions, trials: (
                        print(f"[UTIL] {os.path.basename(input_parquet)} | Type: {plot_type} | Data: {df.shape[0]} points{' | Conditions: ' + str(conditions) if conditions else ''}{(' | Trials: ' + str(trials)) if trials else ''} → {output_path}")
                    ))(df["condition"].unique().to_list() if "condition" in df.columns else [], df["trial_number"].unique().to_list() if "trial_number" in df.columns else [])
                ))(get_output_filename(output_dir, output_prefix))
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
        sys.exit(1)