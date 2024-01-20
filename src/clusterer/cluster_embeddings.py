import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_clusters_DBSCAN(data: list[dict], eps: float=0.5, min_samples: int=10):
    scaler = StandardScaler
    embeddings = [face["embedding"] for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings)
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices = np.where(labels==unique_label)
        for i in indices:
            data[i]["label"] = unique_label
    return data