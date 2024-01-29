import numpy as np
from sklearn.manifold import TSNE

RANDOM_SEED = 42


def transform_tSNE(arr: np.ndarray, n_components: int, perplexity: int) -> np.ndarray:
    """Transform the given ndarray using a tSNE model.

    Args:
        arr: np.ndarray. In this example, an embedding matrix (n by m), where n is the number of examples and m equals to the embedding dimension.
        n_components: int. The number of components for tSNE.
        perplexity: int. Perplexity for tSNE.

    Returns:
        vis_dims: np.ndarray. A transformed matrix of n by n_components.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=RANDOM_SEED, init="random")
    vis_dims = tsne.fit_transform(arr)
    return vis_dims
