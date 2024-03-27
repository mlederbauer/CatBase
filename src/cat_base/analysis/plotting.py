import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

plt.style.use("./style.mplstyle")
plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"


def plot_UMAP(embeddings: pd.DataFrame) -> None:

    # TODO add coloring according to metadata

    # Convert the DataFrame column or list to a numpy array
    embeddings_array = np.stack(embeddings.values)

    # Apply UMAP
    reducer = UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_array)

    # Plotting
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title("UMAP Projection of Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()
