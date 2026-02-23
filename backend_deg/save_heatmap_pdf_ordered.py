import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

def save_heatmap_pdf_ordered(usages_path, metadata, index, out_pdf,
                            width_per_sample=0.20,   # inches per sample
                            height_per_program=0.25, # inches per program row
                            min_w=16, min_h=6, font_size=6):
    """
    Makes a wide vector PDF with sample labels ordered according to metadata.
    
    Parameters:
    usages_path: cNMF usages file (rows=samples, cols=programs)
    metadata: pandas DataFrame with sample ordering information
    out_pdf: where to write the PDF
    """
    df = pd.read_csv(usages_path, sep="\t", index_col=0)   # rows=samples, cols=programs
    
    # Get the sample order from metadata index
    metadata_order = metadata[index].tolist()
    
    # Filter and reorder the usages data to match metadata order
    # Only keep samples that exist in both datasets
    common_samples = [sample for sample in metadata_order if sample in df.index]
    df_ordered = df.loc[common_samples]
    
    data = df_ordered.T  # rows=programs, cols=samples (in metadata order)

    n_samples = data.shape[1]
    n_programs = data.shape[0]

    # Dynamic figure size so labels fit horizontally & vertically
    fig_w = max(min_w, n_samples * width_per_sample)
    fig_h = max(min_h,  n_programs * height_per_program)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(data, cmap="viridis", ax=ax, cbar=True)

    ax.set_title("NMF: Sample vs. Module Weights (W) - Metadata Ordered")
    ax.set_xlabel("samples"); ax.set_ylabel("programs")

    # Force EVERY sample label, no autoskip
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
    
    return common_samples  # Return list of samples that were plotted