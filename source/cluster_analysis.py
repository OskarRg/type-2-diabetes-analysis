import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Performs unsupervised clustering using `KMeans` and provides
    optional dimensionality reduction via `PCA`.

    The class is intended to support exploratory analysis such as
    segmenting patients into groups with similar medical profiles.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        """
        Initialize the analyzer.

        :param n_clusters: Number of clusters for KMeans.
        :param random_state: Random seed for reproducibility.
        """
        self.n_clusters: int = n_clusters
        self.random_state: int = random_state
        self.model: KMeans | None = None
        self.pca: PCA | None = None
        self.labels: np.ndarray | None = None
        self.pca_coords: np.ndarray | None = None

    def fit_predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the `KMeans` model and return cluster labels.

        Also performs 2-dimensional PCA projection for visualization.

        :param X: Input features as a DataFrame.
        :return: Tuple `(labels, pca_coords)` where:
        """
        logger.debug("Fitting KMeans clustering…")

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.labels: np.ndarray = self.model.fit_predict(X)
        self.pca = PCA(n_components=2, random_state=self.random_state)
        self.pca_coords: np.ndarray = self.pca.fit_transform(X)

        return self.labels, self.pca_coords

    def plot_clusters(
        self,
        title: str = "KMeans clusters (PCA projection)",
        show_figure: bool = True,
        save_path: Path | None = None,
    ) -> None:
        """
        Visualize clusters using the 2D PCA projection.

        :param labels: `np.ndarray` of cluster assignments.
        :param pca_coords: `np.ndarray` of shape `(n_samples, 2)` containing PCA coordinates.
        :param title: Plot title.
        """
        if self.pca_coords.shape[1] != 2:
            raise ValueError("PCA coordinates must be 2-dimensional for plotting.")

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            self.pca_coords[:, 0],
            self.pca_coords[:, 1],
            c=self.labels,
            cmap="viridis",
            alpha=0.7,
            s=40,
        )

        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(alpha=0.3)

        if show_figure:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.debug(f"Plot saved to: {save_path}")

    def plot_cluster_stats(
        self,
        X_test: pd.DataFrame,
        title: str | None = "Feature means by cluster",
        show_figure: bool = True,
        save_path: Path | None = None,
    ) -> None:
        """
        Calculates the mean values of features for each cluster and displays a heatmap.

        This function appends the cluster labels to the feature DataFrame, groups the data
        by cluster, computes the mean for each feature, and visualizes these means using
        a Seaborn heatmap.

        :param X_test: A pandas DataFrame containing the features (test data).
        :param labels: A numpy array of cluster labels assigned to the samples in X_test.
        """
        df_clusters: pd.DataFrame = X_test.copy()
        df_clusters["cluster"] = self.labels
        cluster_stats: pd.DataFrame = df_clusters.groupby("cluster").mean()

        plt.figure(figsize=(10, 5))
        sns.heatmap(cluster_stats, cmap="viridis", annot=True)
        plt.title(title)
        if show_figure:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.debug(f"Plot saved to: {save_path}")
