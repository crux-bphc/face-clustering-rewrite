import pickle
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_clusters_DBSCAN(data: list[dict], eps: float=0.5, min_samples: int=10, scale: bool=True):
    scaler = StandardScaler
    embeddings = [face["embedding"] for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices = np.where(labels==unique_label)
        for i in indices:
            data[i]["label"] = unique_label
    return data, labels, unique_labels

def create_clusters_Agglomerative(data: list[dict], n_clusters: int=None, distance_threshold: float=0.5,
                                  scale: bool=True):
    scaler = StandardScaler
    embeddings = [face["embedding"] for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices = np.where(labels==unique_label)
        for i in indices:
            data[i]["label"] = unique_label
    return data, labels, unique_labels

def create_clusters_KMeans(data: list[dict], n_clusters: int=8, scale: bool=True):
    scaler = StandardScaler
    embeddings = [face["embedding"] for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices = np.where(labels==unique_label)
        for i in indices:
            data[i]["label"] = unique_label
    return data, labels, unique_labels