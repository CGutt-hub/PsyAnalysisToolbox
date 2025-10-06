import polars as pl, matplotlib.pyplot as plt, sys, os
from matplotlib.backends.backend_pdf import PdfPages
if __name__ == "__main__":
    usage = lambda: print("[Nextflow] Usage: python plotter.py <plot_type> <input_parquet> <output_dir>") or sys.exit(1)
    get_output_filename = lambda input_file, plot_type: f"{os.path.splitext(os.path.basename(input_file))[0]}_{plot_type}.pdf"
    get_attachment_filename = lambda input_file, plot_type: f"{os.path.splitext(os.path.basename(input_file))[0]}_{plot_type}_data.parquet"
    run = lambda plot_type, input_parquet, output_dir: (
        print(f"[Nextflow] Plotting started for: {input_parquet} as {plot_type}") or
        (lambda df:
            print(f"[Nextflow] Parquet file loaded: {input_parquet}, shape: {df.shape}") or
            (lambda fig:
                print(f"[Nextflow] Figure created for plot type: {plot_type}") or
                (lambda ax:
                    print(f"[Nextflow] Axes created, plotting data...") or
                    ax.set_facecolor('white') or
                    plt.style.use('grayscale') or
                    print(f"[Nextflow] Plot style set to greyscale.") or
                    print(f"[Nextflow] {'Scatter' if plot_type == 'scatter' else 'Bar' if plot_type == 'bar' else 'Line'} plot selected.") or
                    (ax.scatter(df[df.columns[0]], df[df.columns[1]], color='black') if plot_type == 'scatter' else ax.bar(df[df.columns[0]], df[df.columns[1]], color='grey') if plot_type == 'bar' else ax.plot(df[df.columns[0]], df[df.columns[1]], color='black')) or
                    print(f"[Nextflow] Data plotted.") or
                    ax.set_title(f"{plot_type.title()} Plot") or
                    print(f"[Nextflow] Title set: {plot_type.title()} Plot") or
                    ax.set_xlabel(df.columns[0]) or
                    print(f"[Nextflow] X label set: {df.columns[0]}") or
                    ax.set_ylabel(df.columns[1]) or
                    print(f"[Nextflow] Y label set: {df.columns[1]}") or
                    plt.tight_layout() or
                    print(f"[Nextflow] Layout tightened.") or
                    (lambda pdf_path: (
                        print(f"[Nextflow] Saving plot to {pdf_path}") or
                        PdfPages(pdf_path).savefig(fig) or
                        print(f"[Nextflow] Plot PDF/A saved: {pdf_path}")
                    ))(os.path.join(output_dir, get_output_filename(input_parquet, plot_type))) or
                    (lambda att_path: (
                        print(f"[Nextflow] Saving parquet attachment to {att_path}") or
                        df.write_parquet(att_path) or
                        print(f"[Nextflow] Parquet attachment saved: {att_path}")
                    ))(os.path.join(output_dir, get_attachment_filename(input_parquet, plot_type)))
                )(fig.add_subplot(111))
            )(plt.figure())
        )(pl.read_parquet(input_parquet))
    )
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
