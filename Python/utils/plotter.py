import polars as pl, matplotlib.pyplot as plt, matplotlib, sys, os, shutil
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')

if __name__ == "__main__":
    usage = lambda: print("Usage: python plotter.py <input_parquet> <output_dir>") or sys.exit(1)
    get_output_filename = lambda input_file, output_dir: os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_plot.pdf")
    
    # Working plotting function
    create_plot = lambda df, plot_type, output_path, input_parquet: (
        print(f"[Nextflow] Creating {plot_type} plot with {df.height} data points"),
        plt.style.use('grayscale'),
        (lambda y_data, fig_ax_tuple: (
            (lambda fig, ax: (
                # Create the appropriate plot
                plot_type == "bar" and (
                    # Handle questionnaire data properly
                    "x_axis" in df.columns and "y_axis" in df.columns and (
                        (lambda x_labels, y_values, y_labels: (
                            ax.bar(range(len(x_labels)), y_values, color='gray', alpha=0.7),
                            ax.set_xticks(range(len(x_labels))),
                            ax.set_xticklabels([label[:50] + "..." if len(label) > 50 else label for label in x_labels], 
                                             rotation=45, ha='right', fontsize=8),
                            # Only add axis labels for non-self-descriptive scales (continuous/numbered data)
                            "x_scale" in df.columns and df["x_scale"][0] == "nominal" and "x_label" in df.columns and ax.set_xlabel(df["x_label"][0]),
                            "y_scale" in df.columns and df["y_scale"][0] == "nominal" and "y_label" in df.columns and ax.set_ylabel(df["y_label"][0]),
                            # Add y-axis labels if available
                            y_labels and any(label for label in y_labels) and (
                                ax.set_yticks(y_values),
                                ax.set_yticklabels([label if label else str(val) for label, val in zip(y_labels, y_values)], 
                                                 fontsize=8)
                            ),
                            print(f"[Nextflow] Questionnaire bar chart created with {len(x_labels)} items")
                        )[-1])(
                            df["x_axis"].to_list(),
                            df["y_data"].to_list(),
                            df["y_axis"].to_list() if "y_axis" in df.columns else []
                        )
                    ) or (
                        # Fallback to simple value counts for non-questionnaire data
                        (lambda value_counts: (
                            ax.bar([str(val) for val in value_counts[""].to_list()], 
                                  value_counts["count"].to_list(), 
                                  color='gray', alpha=0.7),
                            ax.set_xlabel("Values"),
                            ax.set_ylabel("Count"),
                            print(f"[Nextflow] Simple bar chart created with {len(value_counts)} bars")
                        )[-1])(pl.Series(y_data).value_counts().sort(pl.col("").cast(pl.String)))
                    )
                ) or
                plot_type == "line" and (
                    ax.plot(range(len(y_data)), y_data, color='black', linewidth=2, marker='o', markersize=4),
                    ax.set_xlabel("Index"),
                    ax.set_ylabel("Values"),
                    print(f"[Nextflow] Line plot created with {len(y_data)} points")
                )[-1] or
                plot_type == "scatter" and (
                    ax.scatter(range(len(y_data)), y_data, color='gray', alpha=0.6, s=50),
                    ax.set_xlabel("Index"),
                    ax.set_ylabel("Values"),
                    print(f"[Nextflow] Scatter plot created with {len(y_data)} points")
                )[-1],
                # Apply styling
                ax.grid(True, alpha=0.3),
                ax.spines['top'].set_visible(False),
                ax.spines['right'].set_visible(False),
                plt.tight_layout(),
                # Save with metadata
                print(f"[Nextflow] Saving plot to: {output_path}"),
                (lambda pdf_pages: (
                    pdf_pages.savefig(fig, bbox_inches='tight', dpi=300,
                                    metadata={'Title': f'Analysis Plot from {os.path.basename(input_parquet)}',
                                             'Subject': f'Generated from parquet data: {input_parquet}',
                                             'Creator': 'EmotiView Analysis Pipeline'}),
                    pdf_pages.close()
                ))(PdfPages(output_path)),
                plt.close(),
                print(f"[Nextflow] Plot saved: {output_path}")
            )[-1])(fig_ax_tuple[0], fig_ax_tuple[1])
        ))(
            df["y_data"].to_list() if "y_data" in df.columns else [1] * df.height,
            plt.subplots(figsize=(12, 8))  # Larger figure for questionnaire plots
        )
    )[-1]
    
    run = lambda input_parquet, output_dir: (
        print(f"[Nextflow] Generic plotting started for: {input_parquet}"),
        (lambda df: (
            print(f"[Nextflow] Data loaded, shape: {df.shape}"),
            print(f"[Nextflow] Columns: {df.columns}"),
            os.makedirs(output_dir, exist_ok=True),
            (lambda output_path:
                "plot_type" in df.columns and 
                create_plot(df, df["plot_type"][0], output_path, input_parquet) or
                print("[Nextflow] Error: Input parquet file must contain 'plot_type' column")
            )(get_output_filename(input_parquet, output_dir))
        )[-1])(pl.read_parquet(input_parquet))
    )[-1]
    
    try:
        args = sys.argv
        if len(args) < 3:
            usage()
        else:
            run(args[1], args[2])
    except Exception as e:
        print(f"[Nextflow] Plotting errored for input: {sys.argv[1] if len(sys.argv)>1 else 'UNKNOWN'}. Error: {e}")
        sys.exit(1)