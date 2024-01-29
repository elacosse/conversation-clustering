from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from conversation_clustering import clustering


def plot_embeddings(
    embed_arr_ls: List[np.ndarray], n_components: int, names: List[str], perplexity: int, save_path: str
):
    """Plot transformed embedding vectors with predefined labels.

    Args:
        embed_arr_ls: a list of np.ndarray. Each np.ndarray is a matrix with embeddings corresponding to data examples.
        n_components: int. The number of components for tSNE.
        names: a list of str. The names of the data sources. The length of this list should be the same as the length of embed_arr_ls.
        save_path: path to save figure
    Returns:
        None
    """
    vis_dims = clustering.transform_tSNE(np.concatenate(embed_arr_ls), n_components, perplexity)
    colors = ["red", "blue", "green", "orange", "purple"]
    list_names_set = list(set(names))

    colormap = matplotlib.colors.ListedColormap(colors)
    color_indices = []
    # for label in range(len(embed_arr_ls)):
    for label in names:
        color_indices += [list_names_set.index(label)]  # [label] * len(embed_arr_ls[label])
    assert len(vis_dims) == len(color_indices)
    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    # for label in range(len(embed_arr_ls)):
    for color_index in range(len(colors)):
        # color = colors[label]
        color = colors[color_index]
        label_indices = [i for i, value in enumerate(color_indices) if value == color_index]
        avg_x = np.array(x)[label_indices].mean()
        avg_y = np.array(y)[label_indices].mean()
        ax.scatter(avg_x, avg_y, marker="x", color=color, s=100, label=list_names_set[color_index])

    ax.legend()
    plt.title("Conversations sample data visualized in language using t-SNE")
    plt.show()
    plt.savefig(save_path)
