import os
import shutil

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
