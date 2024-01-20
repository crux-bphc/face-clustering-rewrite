import os
import shutil

def create_folder_for_clustering(source_dir: str, dest_dir: str):
    os.makedirs(dest_dir)
    n_files = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(dest_dir, file)

                shutil.move(source_path, destination_path)
                n_files+=1
            if n_files%50==0:
                print(f"Moved {n_files} files!")

