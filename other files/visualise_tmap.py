import os
import pickle
import numpy as np
import tmap as tm
from annoy import AnnoyIndex
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt

# Limit the number of fingerprints to process
take = 300000

# Load fingerprints from pickle file
fingerprints = pickle.load(open('full_v1_combined_FP.pkl', "rb"))[:take]
X = np.array(fingerprints)

# Load reaction input data and extract relevant fields
with open("full_v1_combined.txt", "r", encoding='utf-8') as f:
    tst_input_data = [line.strip() for line in f.readlines()][:take]

rxnes, rxn_classes_full = [], []
for s in tst_input_data:
    temp = s.split('\t')
    rxnes.append(temp[0]) 
    rxn_classes_full.append(temp[1])  # Reaction classes

# Define reaction labels
labels = {
    "1": "Heteroatom alkylation and arylation",
    "2": "Acylation and related processes",
    "3": "C-C bond formation",
    "5": "Protections",
    "6": "Deprotections",
    "7": "Reductions",
    "8": "Oxidations",
    "9": "Functional group interconversion (FGI)",
    "10": "Functional group addition (FGA)",
    '11': 'Our data'
}

# Extract class values for each reaction
y_values = [int(one_rxn.split(".")[0]) - 1 for one_rxn in rxn_classes_full]

# TMAP configuration setup
CFG_TMAP = tm.LayoutConfiguration()
CFG_TMAP.k = 50
CFG_TMAP.kc = 50
CFG_TMAP.sl_scaling_min = 1.0
CFG_TMAP.sl_scaling_max = 1.0
CFG_TMAP.sl_repeats = 1
CFG_TMAP.sl_extra_scaling_steps = 2
CFG_TMAP.placer = tm.Placer.Barycenter
CFG_TMAP.merger = tm.Merger.LocalBiconnected
CFG_TMAP.merger_factor = 2.0
CFG_TMAP.merger_adjustment = 0
CFG_TMAP.fme_iterations = 1000
CFG_TMAP.sl_scaling_type = tm.ScalingType.RelativeToDesiredLength
CFG_TMAP.node_size = 1 / 45
CFG_TMAP.mmm_repeats = 1

def main():
    """Main function to compute nearest neighbors and generate a visualization."""
    print('Started KNN computation...')
    dims = 2048  # Dimensionality of fingerprints

    # Initialize and build Annoy index
    annoy = AnnoyIndex(dims, metric="angular")
    for i, v in enumerate(X):
        annoy.add_item(i, v)
    annoy.build(10)

    # Generate k-nearest neighbors
    knn = [(i, j, cosine_distance(X[i], X[j])) for i in range(len(X)) for j in annoy.get_nns_by_item(i, 10)]
    print('KNN computation completed.')

    # Layout generation using TMAP
    x, y, s, t, _ = tm.layout_from_edge_list(len(X), knn, config=CFG_TMAP)
    x, y = np.array(x), np.array(y)

    # Plot edges
    for i in range(len(s)):
        plt.plot([x[s[i]], x[t[i]]], [y[s[i]], y[t[i]]], "k-", linewidth=0.3, alpha=0.1, zorder=1)

    # Define color palettes
    palette = [
        "#006ba4", "#ff800e", "#ababab", "#595959", "#5f9ed1",
        "#c85300", "#898989", "#a2c8ec", "#ffbc79", "#cfcfcf"
    ]
    palette2 = ["#7aff7a"]  # Green for special class
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(palette)
    custom_cmap_g = ListedColormap(palette2)

    # Separate green class (class 10) from others for better visualization
    y_values_cmap = np.array(y_values)
    green_mask = (y_values_cmap == 10)
    non_green_mask = (y_values_cmap != 10)

    # Plot green class
    plt.scatter(x[green_mask], y[green_mask], s=0.3, c=y_values_cmap[green_mask], cmap=custom_cmap_g, zorder=1)

    # Plot non-green classes
    plt.scatter(x[non_green_mask], y[non_green_mask], s=0.7, c=y_values_cmap[non_green_mask], cmap=custom_cmap, zorder=2, alpha=1)

    # Remove plot spines and ticks
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])

    # Save plots with and without transparency
    plt.savefig('output.png', dpi=300, transparent=False)
    plt.savefig('output_transparent.png', dpi=300, transparent=True)
    print('Visualization saved.')

if __name__ == "__main__":
    main()
