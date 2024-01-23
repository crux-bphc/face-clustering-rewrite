import numpy as np
from architecture import InceptionResNetV1
from tqdm import tqdm
import os
import tensorflow as tf
from typing import List, Optional
import cv2
from mtcnn import MTCNN
from dataset import DatasetGenerator
import pickle


CWD = os.path.dirname(__file__)
IMAGE_SIZE = (160, 160)

class Facenet:

    """
    Pipeline for training and inference using Facenet model to generate face embeddings.
    • To train model call the fit method by passing the directory subdirectories containing images of each person.
    • Expected folder structure for training: 
                                        Images_Directory
                                            |_Person_1
                                                |_Image_1
                                                |_Image_2
                                                |_Image_3
                                                |_Image_4
                                            |_Person_2
                                                |_Image_1
                                                |_Image_2
                                                |_Image_3
                                                |_Image_4...
    • To generate embeddings all images must be placed in a single directory which should be passed to the predict methods `input_dir` parameter.


    The Facenet architecture is defined in the architecture.py file. This class provides a high-level interface
    for loading, fine-tuning, and using the Facenet model.

    """

    def __init__(self, learning_rate: float = 1e-3, freeze_layers: int = 250, weights_path: str = "weights/model-final.h5"):

        """
        Initialize a Facenet instance.

        Args:
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            freeze_layers (int, optional): Number of layers of the pre-trained model to freeze. Defaults to 275.
            default_weights (bool, optional): Whether to use the default weights or fine-tuned weights. Defaults to True.
        """
        super(Facenet, self).__init__()

        # Create the base InceptionResNetV1 model
        self.model = InceptionResNetV1()

        # Use default or fine-tuned weights

        # Initialize the optimizer with the specified learning rate
        self.optimizer = tf.optimizers.legacy.Adam(learning_rate=learning_rate)

         # Load pre-trained weights
        self.model.load_weights(os.path.join(CWD, weights_path))


        # Set layers as trainable based on freeze_layers
        trainable = False
        i = 0
        for layer in self.model.layers:
            i += 1
            if i == freeze_layers:
                trainable = True
            layer.trainable = trainable

    def _detect_faces(self, detector: MTCNN, 
                 image_path: str) -> Optional[List[np.ndarray]]:
        """
        Detect faces in the given image using MTCNN.

        Args:
            detector (MTCNN): MTCNN face detection model.
            image_path (str): Path to the image.

        Returns:
            list_faces (list of np.ndarray): List of detected face images or None.
        """

        img = cv2.imread(image_path)
        faces = detector.detect_faces(img)
        list_faces = []
        if faces:
            for face in faces:
                x, y, width, height = face['box']
                face_image = img[y : y + height, x : x + width]
                face_image = cv2.resize(face_image, IMAGE_SIZE)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                list_faces.append(face_image)

            return list_faces

        return None

    def _process_images(self, images_dir: str) -> dict:


        detector = MTCNN()
        folders = os.listdir(images_dir)
        face_images = {}
        for folder in folders:
            path = os.path.join(images_dir, folder)
            images = os.listdir(path)
            list_faces = []
            for image in images:
                image_path = os.path.join(path, image)
                faces = self._detect_faces(detector, image_path)
                if faces:
                    for face in faces:
                        list_faces.append(face)
                    face_images[folder] = list_faces
        return face_images



    @tf.function
    def _train_step(self, anchor_batch: tf.Tensor, positive_batch: tf.Tensor, negative_batch: tf.Tensor) -> float:
        """
        Perform a single training step for the Facenet model using triplet loss.

        Args:
            anchor_batch (tf.Tensor): Batch of anchor images.
            positive_batch (tf.Tensor): Batch of positive images.
            negative_batch (tf.Tensor): Batch of negative images.

        Returns:
            loss (float): The computed triplet loss for the current batch.
        """

        with tf.GradientTape() as tape:
            anchor_embeddings = self.model(anchor_batch)
            positive_embeddings = self.model(positive_batch)
            negative_embeddings = self.model(negative_batch)
            loss = self._triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    
    def _triplet_loss(self, anchor_embedding: tf.Tensor, positive_embedding: tf.Tensor, negative_embedding: tf.Tensor, margin: float = 0.2) -> float:
        """
        Calculate the triplet loss for a batch of anchor, positive, and negative embeddings.

        Args:
            anchor_embedding (tf.Tensor): Embeddings of anchor images.
            positive_embedding (tf.Tensor): Embeddings of positive images.
            negative_embedding (tf.Tensor): Embeddings of negative images.
            margin (float, optional): Margin parameter for triplet loss. Defaults to 0.2.

        Returns:
            loss (float): The computed triplet loss.
        """
        pos_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=-1)
        neg_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=-1)
        
        loss = tf.maximum(pos_distance - neg_distance + margin, 0.0)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    
    

    def fit(self, input_dir: str, epochs: int, batch_size: int = 128) -> dict:
        """
        Fine-tune the Facenet model using triplet loss with given input triplets.

        Args:
            input_dir (str): Directory containing image folders.
            epochs (int): The number of training epochs.

        Returns:
            history (dict): A dictionary containing training history (e.g., 'train_loss').
        """
        images_dict = self._process_images(input_dir)
        inputs = DatasetGenerator(images_dict, batch_size = batch_size)

        train_loss = []
        for epoch in range(1, epochs + 1):
            epoch_loss = []
            with tqdm(inputs, unit = "batch") as tepoch:
                for data in tepoch:
                    anchor_batch, positive_batch, negative_batch = data
                    loss = self._train_step(anchor_batch, positive_batch, negative_batch)
                    epoch_loss.append(loss)
            total_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_loss.append(total_epoch_loss)

            print(f"Epoch {epoch}: Loss on train = {total_epoch_loss:.5f}")
            if epoch % 5 == 0:
                self.model.save_weights(os.path.join(CWD, 'weights/model.h5'))

        self.model.save_weights(os.path.join(CWD, os.path.join('weights/model-final.h5')))
        history = {
            'train_loss': train_loss,
        }
        return history

    def predict(self, input_dir: str, verbose: int = 1, embeddings_path: str = "embeddings.pkl", ckpt_num: int = 500) -> List:
        """
        Make predictions using the Facenet model and store embeddings in pickle file.

        Args:
            input_dir (str): Input folder containing images to find embeddings for.
            verbose (int): verbose value for model.predict
            embeddings_path (str): Path of pickle file to save embeddings.

        Returns:
            image_embeddings (list) : List of dictonaries with each dictionary containing image path as key and embedding as value.
        """

        images = os.listdir(input_dir)
        print(len(images))
        image_count = 0
        detector = MTCNN()
        image_embeddings = []
        for image in images:
            image_path = os.path.join(input_dir, image)
            faces = self._detect_faces(detector, image_path)
            if faces:
                for face in faces:
                    face = np.expand_dims(face, axis = 0)
                    face = (face - 127.5) / 128.0
                    embeddings = self.model.predict(face, verbose = verbose)
                    image_embeddings.append({"filepath": image, "embedding": embeddings})
            image_count += 1
            if image_count % ckpt_num == 0:
                if os.path.exists(embeddings_path) and os.path.getsize(embeddings_path) > 0:
                    with open(embeddings_path, 'rb') as f:
                        existing_embeddings = pickle.load(f)
                else:
                    existing_embeddings = []

                existing_embeddings.extend(image_embeddings)

                with open(embeddings_path, 'wb') as f:
                    pickle.dump(existing_embeddings, f)
                print(f"{image_count} images done!")

                image_embeddings = []
        
        if image_embeddings:
            if os.path.exists(embeddings_path) and os.path.getsize(embeddings_path) > 0:
                with open(embeddings_path, 'rb') as f:
                    existing_embeddings = pickle.load(f)
            else:
                existing_embeddings = []

            existing_embeddings.extend(image_embeddings)

            with open(embeddings_path, 'wb') as f:
                pickle.dump(existing_embeddings, f)

        return existing_embeddings
    
    def summary(self):
        """
        Display a summary of the Facenet model's architecture.

        Returns:
            str: Model summary.
        """
        return self.model.summary()


if __name__ == "__main__":
    model = Facenet()
    model.summary()

