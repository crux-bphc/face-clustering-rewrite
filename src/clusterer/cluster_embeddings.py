import pickle
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

def create_clusters_DBSCAN(data: list[dict], eps: float=0.5, min_samples: int=10, scale: bool=True, metric: str="euclidean"):
    scaler = StandardScaler()
    embeddings = [tf.squeeze(face["embedding"]) for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    cluster_attributes = {}
    for unique_label in unique_labels:
        cluster_embeddings = []
        indices = np.where(labels==unique_label)[0]
        for i in indices:
            data[i]["label"] = unique_label
            cluster_embeddings.append(data[i]["embedding"])
        cluster_attributes[unique_label] = [np.mean(cluster_embeddings, axis=0), np.std(cluster_embeddings, axis=0)]
    return data, labels, unique_labels, cluster_attributes

def create_clusters_Agglomerative(data: list[dict], n_clusters: int=None, distance_threshold: float=0.5,
                                  scale: bool=True, metric: str="euclidean"):
    scaler = StandardScaler()
    embeddings = [tf.squeeze(face["embedding"]) for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, metric=metric)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    cluster_attributes = {}
    for unique_label in unique_labels:
        cluster_embeddings = []
        indices = np.where(labels==unique_label)[0]
        for i in indices:
            data[i]["label"] = unique_label
            cluster_embeddings.append(data[i]["embedding"])
        cluster_attributes[unique_label] = [np.mean(cluster_embeddings, axis=0), np.std(cluster_embeddings, axis=0)]
    return data, labels, unique_labels, cluster_attributes

def create_clusters_KMeans(data: list[dict], n_clusters: int=8, scale: bool=True, metric: str="euclidean"):
    scaler = StandardScaler()
    embeddings = [tf.squeeze(face["embedding"]) for face in data]
    scaler.fit(embeddings)
    scaled_embeddings = scaler.transform(embeddings) if scale else embeddings
    clusterer = KMeans(n_clusters=n_clusters, metric=metric)
    clusterer.fit(scaled_embeddings)
    labels = clusterer.labels_
    unique_labels = np.unique(labels)
    cluster_attributes = {}
    for unique_label in unique_labels:
        cluster_embeddings = []
        indices = np.where(labels==unique_label)[0]
        for i in indices:
            data[i]["label"] = unique_label
            cluster_embeddings.append(data[i]["embedding"])
        cluster_attributes[unique_label] = [np.mean(cluster_embeddings, axis=0), np.std(cluster_embeddings, axis=0)]
    return data, labels, unique_labels, cluster_attributes
