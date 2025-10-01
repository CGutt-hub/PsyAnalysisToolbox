import polars as pl, matplotlib.pyplot as plt, sys, os
from matplotlib.backends.backend_pdf import PdfPages
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python plotter.py <plot_type> <input_parquet> <output_dir>") or sys.exit(1)
    run = lambda plot_type, input_parquet, output_dir: (
        print(f"[Nextflow] Plotting started for: {input_parquet} as {plot_type}") or None,
        (lambda df: (
            print(f"[Nextflow] Parquet file loaded: {input_parquet}, shape: {df.shape}"),
            (lambda fig: (
                print(f"[Nextflow] Figure created for plot type: {plot_type}"),
                (lambda ax: (
                    print(f"[Nextflow] Axes created, plotting data..."),
                    ax.set_facecolor('white'),
                    plt.style.use('grayscale'),
                    print(f"[Nextflow] Plot style set to greyscale."),
                    print(f"[Nextflow] {'Scatter' if plot_type == 'scatter' else 'Bar' if plot_type == 'bar' else 'Line'} plot selected."),
                    (
                        ax.scatter(df[df.columns[0]], df[df.columns[1]], color='black') if plot_type == 'scatter' else
                        ax.bar(df[df.columns[0]], df[df.columns[1]], color='grey') if plot_type == 'bar' else
                        ax.plot(df[df.columns[0]], df[df.columns[1]], color='black')
                    ),
                    print(f"[Nextflow] Data plotted."),
                    ax.set_title(f"{plot_type.title()} Plot"),
                    print(f"[Nextflow] Title set: {plot_type.title()} Plot"),
                    ax.set_xlabel(df.columns[0]),
                    print(f"[Nextflow] X label set: {df.columns[0]}"),
                    ax.set_ylabel(df.columns[1]),
                    print(f"[Nextflow] Y label set: {df.columns[1]}"),
                    plt.tight_layout(),
                    print(f"[Nextflow] Layout tightened."),
                    (lambda pdf_path: (
                        print(f"[Nextflow] Saving plot to {pdf_path}"),
                        PdfPages(pdf_path).savefig(fig),
                        print(f"[Nextflow] Plot PDF/A saved: {pdf_path}")
                    ))(os.path.join(output_dir, f"plot_{plot_type}.pdf")),
                    (lambda att_path: (
                        print(f"[Nextflow] Saving parquet attachment to {att_path}"),
                        df.write_parquet(att_path),
                        print(f"[Nextflow] Parquet attachment saved: {att_path}")
                    ))(os.path.join(output_dir, f"plot_{plot_type}_data.parquet"))
                ))(fig.add_subplot(111)),
            ))(plt.figure()),
        ))(pl.read_parquet(input_parquet))
    )
    try:
        args = sys.argv
        (lambda a: usage() if len(a) < 4 else run(a[1], a[2], a[3]))(args)
    except Exception as e:
        print(f"[Nextflow] Plotting errored. Error: {e}")
        sys.exit(1)
