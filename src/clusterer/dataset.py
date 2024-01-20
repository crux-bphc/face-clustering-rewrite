import os
import random
import numpy as np
import cv2
from typing import List, Optional, Tuple
os.environ["TF_CPP_MIN_LOG"] = "2"

import tensorflow as tf
CWD = os.path.dirname(__file__)
IMAGE_SIZE = (160, 160)

class DatasetGenerator(tf.keras.utils.Sequence):

    """
    Class that generates batches of (anchor, positive, negative) triplets after preprocessing the images.
    """

    def __init__(self, images_dict: dict, batch_size: int = 128):
        """
        Initialize a DatasetGenerator instance.

        Args:  
            images_dict (dict): Dictionary containing list of detected faces for each folder.
            batch_size (int, optional): Size of the batch generated. Defaults to 128.
        """
        super(DatasetGenerator, self).__init__()
        self.batch_size = batch_size
        self.images_dict = images_dict
        self.triplets = self._create_triplets()
    
    def _create_triplets(self, max_files: int = 10) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        triplets = []
        folders = list(self.images_dict.keys())

        for folder in folders:
            images = self.images_dict[folder][:max_files]
            num_imgs = len(images)
            if num_imgs == 1:
                continue

            for i in range(num_imgs - 1):
                for j in range(i + 1, num_imgs):
                    anchor = (images[i] - 127.5) / 128.0
                    positive = (images[j] - 127.5) / 128.0
                    neg_folder = folder
                    while neg_folder == folder:
                        neg_folder = random.choice(folders)
                    neg_file_index = random.randint(0, len(self.images_dict[neg_folder]) - 1)
                    negative = (self.images_dict[neg_folder][neg_file_index] - 127.5) / 128.0

                    triplets.append((anchor, positive, negative))

        random.shuffle(triplets)
        return triplets

    
    def __len__(self) -> int:

        """
        Returns the number of batches.

        Returns:
            int: Number of batches.
        """

        return int(np.ceil(len(self.triplets) / self.batch_size))
    
    def __getitem__(self, index: int) -> List[np.ndarray]:

        """
        Generates a batch of data.

        Args:
            index (int): Index of the batch to generate.

        Returns:
            List: A List containing three NumPy arrays (anchors, positives, negatives).
        """
        batches = self.triplets[index * self.batch_size:(index + 1) * self.batch_size]
        anchors = []
        positives = []
        negatives = []
        for triplet in batches:
            anchor, positive, negative = triplet
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        anchors = np.array(anchors)
        positives = np.array(positives)
        negatives = np.array(negatives)


        return ([anchors, positives, negatives])

