import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

def save_heatmap_pdf_all_labels(usages_path, out_pdf,
                                width_per_sample=0.20,   # inches per sample
                                height_per_program=0.25, # inches per program row
                                min_w=16, min_h=6, font_size=6):
    """
    Makes a very wide vector PDF that includes ALL sample labels.
    usages_path: cNMF usages (rows=samples, cols=programs)
    out_pdf: where to write the PDF
    """
    df = pd.read_csv(usages_path, sep="\t", index_col=0)   # rows=samples, cols=programs
    data = df.T                                            # rows=programs, cols=samples

    n_samples = data.shape[1]
    n_programs = data.shape[0]

    # Dynamic figure size so labels fit horizontally & vertically
    fig_w = max(min_w, n_samples * width_per_sample)
    fig_h = max(min_h,  n_programs * height_per_program)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(data, cmap="viridis", ax=ax, cbar=True)

    ax.set_title("NMF: Sample vs. Module Weights (W)")
    ax.set_xlabel("samples"); ax.set_ylabel("programs")

    # --- Force EVERY sample label, no autoskip ---
    # Heatmap cells are centered at x = 0..n-1; using +0.5 matches Seaborn's grid centers.
    positions = np.arange(n_samples) + 0.5
    ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(list(data.columns)))

    # Styling for readability
    ax.tick_params(axis="x", which="major", rotation=90, labelsize=font_size, length=0)
    ax.tick_params(axis="y", labelsize=font_size)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")  # vector PDF with all labels
    plt.close(fig)
