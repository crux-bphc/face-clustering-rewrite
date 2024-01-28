import os
import shutil
import numpy as np

def create_folder_for_clustering(source_dir: str, dest_dir: str):
    os.makedirs(dest_dir)
    n_files = 0
    for root, dirs, files in os.walk(source_dir):
        for dir in dirs:
            for file in os.listdir(os.path.join(root,dir)):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(root,dir,file)
                    newfile = dir+"-img-"+file
                    destination_path = os.path.join(dest_dir, newfile)
                    shutil.copy(source_path, destination_path)
                    n_files+=1
                if n_files%100==0:
                    print(f"Copied {n_files} files!")

def create_person_attributes(data: list[dict]):
    sorted_data = sorted(data, key= lambda x: x["filepath"])
    person_embeddings = {}
    current_person = sorted_data[0]["filepath"].split("-img-")[0]
    person_embeddings[current_person] = []
    for datapoint in sorted_data:
        person = datapoint["filepath"].split("-img-")[0]
        if person != current_person:
            current_person = person
            person_embeddings[current_person] = []
        person_embeddings[current_person].append(datapoint["embedding"])
    person_attributes = {}
    for person in person_embeddings.keys():
        person_attributes[person] = [np.mean(person_embeddings[person], axis=0), np.std(person_embeddings[person], axis=0)]
    return person_attributes
