import os
import math

def pairwise_metrics(persons_dict: dict, clusters_dict: dict, images_folder: str):
    no_pairs = {}
    for image in os.listdir(images_folder):
        if (image.split('-img-')[0] in no_pairs.keys()):
            no_pairs[image.split('-img-')[0]] += 1
        else:
            no_pairs[image.split('-img-')[0]] = 1
    for person in no_pairs:
        no_pairs[person] = math.comb(no_pairs[person], 2)
    
    correctly_clustered_pairs = {}
    for person in persons_dict:
        person_clusters = set(persons_dict[person])
        for cluster in person_clusters:
            count = persons_dict[person].count(cluster)
            if (person in correctly_clustered_pairs.keys()):
                correctly_clustered_pairs[person] += math.comb(count, 2)
            else:
                correctly_clustered_pairs[person] = math.comb(count, 2)

    cluster_pairs = {}
    for cluster in clusters_dict:
        cluster_pairs[cluster] = math.comb(clusters_dict[cluster][-1], 2)

    results = {}
        
    precision = sum(correctly_clustered_pairs.values()) / sum(no_pairs.values())
    results['precision'] = precision

    recall = sum(correctly_clustered_pairs.values()) / sum(cluster_pairs.values())
    results['recall'] = recall
    
    f_measure = (2 * precision * recall) / (precision + recall)
    results['f_measure'] = f_measure

    return results
