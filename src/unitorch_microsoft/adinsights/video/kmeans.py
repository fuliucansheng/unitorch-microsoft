from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import os
import time
import fire
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import wandb


class KMeansFamily:
    def __init__(self, model):
        """
        Initialize the KMeans clustering using sklearn's implementation.

        Parameters:
        - n_clusters: Number of clusters
        - max_iter: Maximum number of iterations
        - tol: Tolerance for convergence
        - random_state: Random state for reproducibility
        """
        self.model = model
        self.centroids_indices = []
        self.centroids_members = {}

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X: Input data, a numpy array of shape (n_samples, n_features)
        """
        self.model.fit(X)

    def labels(self):
        """
        Get the labels of the clusters for each sample in X.

        Parameters:
        - X: Input data, a numpy array of shape (n_samples, n_features)

        Returns:
        - labels: Cluster labels for each sample
        """
        return self.model.labels_

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters:
        - X: Input data, a numpy array of shape (n_samples, n_features)

        Returns:
        - labels: Cluster labels for each sample
        """
        return self.model.predict(X)

    def get_centroids(self):
        """
        Get the centroids of the clusters.

        Returns:
        - centroids: A numpy array of shape (n_clusters, n_features)
        """
        return self.model.cluster_centers_

    def set_centroids_members(self):
        labels = self.labels()
        for index, label in enumerate(labels):
            if label not in self.centroids_members:
                self.centroids_members[label] = []
            self.centroids_members[label].append(index)

    def sample_within_clusters(self, cluster_id, k=5):
        members = self.centroids_members[cluster_id]
        samples = np.random.choice(members, size=min(k, len(members)), replace=False)
        return samples

    def set_centroids_indices(self, X):
        centroids = self.get_centroids()
        labels = self.labels()
        self.centroids_indices = [-1 for i in range(len(centroids))]
        print(f"check center  indices {self.centroids_indices}")
        exact_cnt = 0
        approx_cnt = 0
        valid_cnt = 0
        for cindex, centroid in enumerate(centroids):
            for index, data in enumerate(X):
                if np.array_equal(centroid, data):
                    exact_cnt += 1
                    approx_cnt += 1
                    self.centroids_indices[cindex] = index
                    if labels[index] == cindex:
                        valid_cnt += 1
                    break
                if np.allclose(centroid, data, rtol=1e-5, atol=1e-8):
                    approx_cnt += 1
                    self.centroids_indices[cindex] = index
                    if labels[index] == cindex:
                        valid_cnt += 1
                    # print(f"check equal {centroid} {data}")
                    # print(f"check {cindex} {labels[index]} {self.centroids_indices[cindex]}")
                    break
        print(f"after check center  indices {self.centroids_indices}")
        with_label_cnt = np.count_nonzero(self.centroids_indices != -1)
        print(
            f"Centroids shape: {centroids.shape} exact_cnt {exact_cnt} approx_cnt {approx_cnt} centroids found label {with_label_cnt} valid label setting {valid_cnt}"
        )

    def inertia(self):
        """
        Get the inertia of the model.

        Returns:
        - inertia: The sum of squared distances to the nearest cluster center
        """
        return self.model.inertia_

    def evaluate(self, X):
        """
        Evaluate the model using silhouette score.

        Parameters:
        - X: Input data, a numpy array of shape (n_samples, n_features)

        Returns:
        - silhouette_score: The silhouette score of the clustering
        """
        labels = self.labels()
        centroids = self.get_centroids()
        inertia = self.inertia()
        sil_coeff = silhouette_score(X, labels, metric="euclidean")
        cluster_sizes = {i: np.sum(labels == i) for i in range(len(centroids))}
        total_clusters = len(cluster_sizes)
        max_size_cluster = max(cluster_sizes.values())
        min_size_cluster = min(cluster_sizes.values())
        avg_size_cluster = np.mean(list(cluster_sizes.values()))
        list_cluster_size = list(cluster_sizes.values())
        np.sort(list_cluster_size)
        print(list_cluster_size)

        print("====== Evaluate =======")
        print(f"Centroids shape: {centroids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(
            f"Total Clusters: {total_clusters}; max_size_cluster: {max_size_cluster}; min_size_cluster: {min_size_cluster}; avg_size_cluster: {avg_size_cluster:.2f}"
        )

        print(f"Inertia: {inertia:.2f}; normalized inertia: {inertia / X.shape[0]:.2f}")
        print(f"Silhouette Coefficient: {sil_coeff:.3f}")
        print("====== End Evaluate =======")
        return

    def visualize(self, X, output_file):
        """
        Visualize the clustering results.

        Parameters:
        - X: Input data, a numpy array of shape (n_samples, n_features)
        - labels: Cluster labels for each sample
        - centroids: Centroids of the clusters
        - metas: Meta information for each sample
        - output_file: Path to the output file for visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Visualization code goes here
        # For example, using t-SNE to reduce dimensionality and plot the clusters
        centroids = self.get_centroids()
        X_size = X.shape[0]
        centroids_size = centroids.shape[0]
        total_emb = np.concatenate((X, centroids), axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(total_emb)
        print(f"input x size {X.shape} {X_embedded.shape}")
        labels = self.labels()
        print(f"labels shape {labels.shape}")
        plt.figure(figsize=(16, 9))
        colors = [i for i in range(len(centroids))]

        """
        for label,embed in zip(labels, X_embedded[:X_size]):
            x = embed[0]
            y = embed[1]
            plt.scatter(x, y, alpha=0.7, c=colors[label], cmap='viridis')
        """
        plt.scatter(
            X_embedded[:X_size, 0],
            X_embedded[:X_size, 1],
            alpha=0.7,
            c=labels,
            cmap="viridis",
        )
        plt.scatter(
            X_embedded[X_size:, 0],
            X_embedded[X_size:, 1],
            c="red",
            marker="x",
            s=100,
            label="Centroids",
        )
        for cid in range(len(centroids)):
            plt.annotate(
                str(cid),
                alpha=0.5,
                xy=(X_embedded[X_size + cid, 0], X_embedded[X_size + cid, 1]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=12,
            )
        print("finish")
        plt.colorbar()
        # plt.legend(loc=4)
        plt.grid(True)
        # plt.show()
        plt.savefig(output_file)
        plt.close()

        # Plotting code goes here
        # For example, using matplotlib to create a scatter plot
        # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
        # plt.scatter(X_embedded[centroids[:, 0], X_embedded[:, 1]], c='red', marker='x', s=100, label='Centroids')
        # plt.title('KMeans Clustering')
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        # plt.legend()
        # plt.savefig(output_file)
        # plt.show()
        # Save the visualization to a file
        # plt.savefig(output_file)
        # plt.close()
        # https://colab.research.google.com/github/practical-nlp/practical-nlp/blob/master/Ch3/09_Visualizing_Embeddings_Using_TSNE.ipynb?authuser=0&pli=1#scrollTo=LtOVUzKuhfAu
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE


def model_factory(model_name: str, **kwargs) -> KMeansFamily:
    """
    Factory function to create a KMeans model based on the model name.

    Parameters:
    - model_name: Name of the model to create (e.g., "kmeans", "minibatchkmeans", "bisectingkmeans")
    - kwargs: Additional parameters for the model

    Returns:
    - model: An instance of the KMeansFamily class
    """
    if model_name == "kmeans":
        model = KMeans(
            n_clusters=kwargs["n_clusters"],
            max_iter=kwargs["max_iter"],
            tol=kwargs.get("tol", 1e-4),
            random_state=kwargs.get("random_state", None),
        )
        return KMeansFamily(model)
    elif model_name == "minibatchkmeans":
        model = MiniBatchKMeans(
            n_clusters=kwargs["n_clusters"],
            max_iter=kwargs["max_iter"],
            batch_size=kwargs.get("batch_size", 1024),
        )
        return KMeansFamily(model)
    elif model_name == "bisectingkmeans":
        model = BisectingKMeans(
            n_clusters=kwargs["n_clusters"],
            max_iter=kwargs["max_iter"],
            tol=kwargs.get("tol", 1e-4),
            random_state=kwargs.get("random_state", None),
        )
        return KMeansFamily(model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def wandb_login(wandb_entity):
    """
    Login to Weights & Biases (wandb) using the API key from the environment variable.
    """
    if "WANDB_API_KEY" in os.environ:
        try:
            assert wandb_entity is not None, "wandb entity is None"
            wandb.login(key=os.getenv("WANDB_API_KEY"),relogin=True)
            current_time = time.strftime("%m/%d/%Y/%H", time.localtime())
            wandb.init(
                project="kmeans",
                entity=wandb_entity,
                name=f"kmeans_{current_time}",
            )
            print( "wandb login success")
        except:
            print("Failed to login to wandb. Please check your API key.")
            return False
    else:
        print("WANDB_API_KEY not found in environment variables. Please set it.")
        return False


def cluster_data(
    data_file: str,
    names: Union[str, List[str]] = "img;emb",
    feature_col: str = "emb",
    meta_col: str = "img",
    model_name: str = "kmeans",
    n_clusters=3,
    max_iter=300,
    batch_size=1024,
    cache_dir: str = "output",
    use_wandb=False,
    wandb_entity=None,
):
    """
    Cluster the data using KMeans.

    Parameters:
    - X: Input data, a numpy array of shape (n_samples, n_features)
    - n_clusters: Number of clusters
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence
    - random_state: Random state for reproducibility

    Returns:
    - labels: Cluster labels for each sample
    - centroids: Centroids of the clusters
    """
    #setup wandb
    use_wandb_flag = False
    if use_wandb:
        use_wandb_flag = wandb_login(wandb_entity)

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]
    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    # features = data[feature_col].values
    features = np.array(
        data[feature_col]
        .apply(lambda x: list(map(float, re.split(r"[ ,;]", x))))
        .tolist()
    )
    metas = data[meta_col].tolist()
    print(f"Data shape: {features.shape}{len(metas)}")
    os.makedirs(cache_dir, exist_ok=True)
    assert feature_col in data.columns, f"Column {feature_col} not found in data."

    kwargs = {
        "n_clusters": n_clusters,
        "max_iter": max_iter,
        "batch_size": batch_size,
    }
    model = model_factory(model_name, **kwargs)

    # fit model
    time_start = time.time()
    model.fit(features)
    model.set_centroids_members()
    time_end = time.time()
    print(f"Model fitting time: {time_end - time_start:.2f} seconds")

    # evaluate
    time_start = time.time()
    model.evaluate(features)
    time_end = time.time()
    print(f"Model evaluation time: {time_end - time_start:.2f} seconds")

    # visualization
    time_start = time.time()
    output_file = os.path.join(cache_dir, f"vis_{model_name}.png")
    model.visualize(features, output_file)
    time_end = time.time()
    print(f"Model visualization time: {time_end - time_start:.2f} seconds")

    # sample for case check
    output_file = f"{cache_dir}/sample.tsv"
    with open(output_file, "w") as writer:
        for i in range(len(model.get_centroids())):
            samples = model.sample_within_clusters(i)
            for sample in samples:
                metainfo = (
                    "/home/lichenshih/VideoProcess/video_processing/" + metas[sample]
                )
                writer.write(str(i) + "\t" + metainfo + "\n")
                writer.flush()
    output_file = f"{cache_dir}/labels.tsv"
    with open(output_file, "w") as writer:
        for meta, label in zip(metas, model.labels()):
            writer.write(str(meta) + "\t" + str(label) + "\n")
            writer.flush()
    output_file = f"{cache_dir}/centroids_emb.tsv"
    with open(output_file, "w") as writer:
        for cnt, center_emb in enumerate(model.get_centroids()):
            center_emb = " ".join(np.array(center_emb, dtype=str))
            writer.write(str(cnt) + "\t" + center_emb + "\n")
            writer.flush()

    return


if __name__ == "__main__":
    fire.Fire()
