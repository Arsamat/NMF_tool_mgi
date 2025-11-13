from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

def plot_clusters(df_original):
    """
    Perform hierarchical clustering on samples (rows of df_original),
    plot dendrogram with matching cluster colors, and return leaf order.
    """
    st.subheader("📊 Hierarchical Clustering of Samples")

    df = df_original.copy()
    # --- Choose number of clusters ---
    num_clusters = st.slider("Select number of clusters", 2, 10, 3)

    # --- Hierarchical clustering ---
    distance_matrix = pdist(df, metric="euclidean")
    Z = linkage(distance_matrix, method="ward")

    # --- Assign cluster labels to samples ---
    cluster_labels = fcluster(Z, num_clusters, criterion="maxclust")
    cluster_labels = pd.Series(cluster_labels, index=df.index)

    # --- Generate distinct cluster colors (as HEX strings) ---
    palette = sns.color_palette("husl", num_clusters)
    color_map = {i + 1: mcolors.to_hex(palette[i]) for i in range(num_clusters)}

    # --- Map leaves (samples) to their cluster colors ---
    leaf_color_map = {i: color_map[cluster_labels[i]] for i in range(len(cluster_labels))}

    # --- Define recursive link color function ---
    def link_color_func(node_id):
        if node_id < len(cluster_labels):
            # leaf node
            return leaf_color_map[node_id]
        else:
            left = int(Z[node_id - len(cluster_labels), 0])
            right = int(Z[node_id - len(cluster_labels), 1])
            left_color = link_color_func(left)
            right_color = link_color_func(right)
            # If both children share a color → keep it; else → gray
            return left_color if left_color == right_color else "#B0B0B0"
    
    # --- Plot dendrogram ---
    fig, ax = plt.subplots(figsize=(14, 5))
    dendro = dendrogram(
        Z,
        labels=None,
        no_labels=True,         # 🚫 No sample labels
        leaf_font_size=0,
        leaf_rotation=90,
        color_threshold=0,
        above_threshold_color="#B0B0B0",
        link_color_func=link_color_func,  # 🎨 color per cluster
        ax=ax,
    )

    # --- Add cluster cutoff line ---
    if num_clusters > 1:
        cutoff = Z[-(num_clusters - 1), 2]
        ax.axhline(y=cutoff, color="black", linestyle="--", linewidth=1.2)

    ax.set_title(f"Hierarchical Clustering Dendrogram (k={num_clusters})", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Distance", fontsize=12)

    # --- Legend showing cluster colors ---
    handles = [plt.Line2D([0], [0], color=color_map[i + 1], lw=6) for i in range(num_clusters)]
    labels = [f"Cluster {i + 1}" for i in range(num_clusters)]
    ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig)

    # --- Return leaf order (sample indices) ---
    leaf_order = dendro["leaves"]
    return leaf_order, cluster_labels


def plot_module_clusters(df_modules):
    """
    Perform hierarchical clustering on modules (rows of df_modules),
    plot a plain dendrogram (gray branches) with colored module labels
    based on cluster assignment. Returns leaf order and cluster labels.
    """
    st.subheader("📊 Hierarchical Clustering of Modules")

    # --- Choose number of clusters ---
    num_clusters = st.slider("Select number of clusters", 2, min(10, len(df_modules)), 3)

    # --- Perform hierarchical clustering ---
    Z = linkage(df_modules.values, method="ward", metric="euclidean")

    # --- Assign cluster labels ---
    cluster_labels = pd.Series(
        fcluster(Z, num_clusters, criterion="maxclust"),
        index=df_modules.index
    )

    # --- Define color palette ---
    palette = sns.color_palette("husl", num_clusters)
    color_map = {i + 1: mcolors.to_hex(palette[i]) for i in range(num_clusters)}

    # --- Plot plain dendrogram (gray branches) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    dendro = dendrogram(
        Z,
        labels=df_modules.index.tolist(),
        color_threshold=0,
        above_threshold_color="#A0A0A0",
        no_labels=False,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax,
    )

    ax.set_title(f"Module Dendrogram (k={num_clusters})", fontsize=14)
    ax.set_ylabel("Distance", fontsize=12)
    ax.set_xlabel("Modules")

    # --- Color module labels (leaf text) by cluster ---
    # dendro["ivl"] gives labels (in leaf order)
    for label_text, module_name in zip(ax.get_xticklabels(), dendro["ivl"]):
        cluster_id = cluster_labels.loc[module_name]
        label_text.set_color(color_map[cluster_id])
        label_text.set_fontweight("bold")

    # --- Add legend ---
    handles = [
        plt.Line2D([0], [0], color=color_map[i + 1], lw=6)
        for i in range(num_clusters)
    ]
    labels = [f"Cluster {i + 1}" for i in range(num_clusters)]
    ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig)

    # --- Return leaf order (list of indices in dendrogram order) and cluster labels ---
    leaf_order = dendro["leaves"]
    return leaf_order, cluster_labels