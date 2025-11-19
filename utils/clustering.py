import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ---------- K-Means Clustering ----------
def run_kmeans(df, n_clusters=3, random_state=42, scale=True):
    """
    Runs K-Means clustering on the data.

    Parameters:
    - df: DataFrame with numeric columns
    - n_clusters: Number of clusters
    - random_state: Random state for reproducibility
    - scale: Whether to standardize features before clustering

    Returns:
    - model: Fitted KMeans model
    - labels: Cluster labels for each sample
    - metrics: Dictionary with evaluation metrics
    """
    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(data_scaled)

    # Calculate evaluation metrics
    metrics = {
        'silhouette_score': silhouette_score(data_scaled, labels),
        'davies_bouldin_score': davies_bouldin_score(data_scaled, labels),
        'calinski_harabasz_score': calinski_harabasz_score(data_scaled, labels),
        'inertia': model.inertia_,
        'n_clusters': n_clusters
    }

    return model, labels, metrics


# ---------- Hierarchical Clustering ----------
def run_hierarchical_clustering(df, n_clusters=3, linkage_method='ward', scale=True):
    """
    Runs Hierarchical (Agglomerative) clustering.

    Parameters:
    - df: DataFrame with numeric columns
    - n_clusters: Number of clusters
    - linkage_method: 'ward', 'complete', 'average', 'single'
    - scale: Whether to standardize features

    Returns:
    - model: Fitted AgglomerativeClustering model
    - labels: Cluster labels
    - linkage_matrix: For dendrogram plotting
    - metrics: Evaluation metrics
    """
    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(data_scaled)

    # Create linkage matrix for dendrogram
    linkage_matrix = linkage(data_scaled, method=linkage_method)

    metrics = {
        'silhouette_score': silhouette_score(data_scaled, labels),
        'davies_bouldin_score': davies_bouldin_score(data_scaled, labels),
        'calinski_harabasz_score': calinski_harabasz_score(data_scaled, labels),
        'n_clusters': n_clusters
    }

    return model, labels, linkage_matrix, metrics


# ---------- DBSCAN Clustering ----------
def run_dbscan(df, eps=0.5, min_samples=5, scale=True):
    """
    Runs DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Parameters:
    - df: DataFrame with numeric columns
    - eps: Maximum distance between two samples for one to be considered in the neighborhood
    - min_samples: Minimum number of samples in a neighborhood for a core point
    - scale: Whether to standardize features

    Returns:
    - model: Fitted DBSCAN model
    - labels: Cluster labels (-1 indicates noise/outliers)
    - metrics: Evaluation metrics
    """
    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'noise_percentage': (n_noise / len(labels)) * 100
    }

    # Only calculate silhouette if we have more than 1 cluster
    if n_clusters > 1:
        # Exclude noise points for silhouette calculation
        if n_noise > 0:
            mask = labels != -1
            if mask.sum() > 0:
                metrics['silhouette_score'] = silhouette_score(data_scaled[mask], labels[mask])
        else:
            metrics['silhouette_score'] = silhouette_score(data_scaled, labels)

    return model, labels, metrics


# ---------- t-SNE Dimensionality Reduction ----------
def run_tsne(df, n_components=2, perplexity=30, random_state=42, scale=True):
    """
    Runs t-SNE for dimensionality reduction and visualization.

    Parameters:
    - df: DataFrame with numeric columns
    - n_components: Number of dimensions (typically 2 or 3)
    - perplexity: Related to number of nearest neighbors (5-50)
    - random_state: Random state for reproducibility
    - scale: Whether to standardize features

    Returns:
    - embedding: t-SNE transformed data
    """
    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    embedding = tsne.fit_transform(data_scaled)

    return embedding


# ---------- UMAP Dimensionality Reduction ----------
def run_umap(df, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, scale=True):
    """
    Runs UMAP for dimensionality reduction and visualization (if available).

    Parameters:
    - df: DataFrame with numeric columns
    - n_components: Number of dimensions (typically 2 or 3)
    - n_neighbors: Size of local neighborhood (2-100)
    - min_dist: Minimum distance between points in low-dimensional space
    - random_state: Random state for reproducibility
    - scale: Whether to standardize features

    Returns:
    - embedding: UMAP transformed data
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")

    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=random_state)
    embedding = umap_model.fit_transform(data_scaled)

    return embedding


# ---------- Elbow Method for K-Means ----------
def calculate_elbow_scores(df, max_k=10, scale=True):
    """
    Calculates inertia scores for different values of k (for elbow method).

    Parameters:
    - df: DataFrame with numeric columns
    - max_k: Maximum number of clusters to test
    - scale: Whether to standardize features

    Returns:
    - scores: Dictionary with k values and corresponding inertia scores
    """
    data = df.copy()

    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values

    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, labels))

    scores = {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

    return scores
