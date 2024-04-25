import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import hyperparameters as hp
from skimage import color

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path

        # Mean and std for standardization
        self.mean = np.zeros((hp.img_size,hp.img_size,3))
        self.std = np.ones((hp.img_size,hp.img_size,3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(os.path.join(self.data_path, "train/"), True)
        self.test_data = self.get_data(os.path.join(self.data_path, "test/"), False)   
             
    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpeg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img


        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)


    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
        # Calculating standardized image.
        img = (img - self.mean) / self.std

        return img

    def custom_preprocess_fn(self, img):
        """ Custom preprocess function for ImageDataGenerator. """

        img = img / 255.
        img = self.standardize(img)
        return img

    def get_data(self, path, shuffle):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"

        Returns:
            An iterable image-batch generator
        """

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                rotation_range=15,
                brightness_range=[0.75,1.0],
                horizontal_flip=True,
                preprocessing_function=self.custom_preprocess_fn
                )

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.img_size, hp.img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle
        )

        lab_data = self.rgb_to_lab(data_gen)
        return lab_data


    def rgb_to_lab(self, data):
        """ Converts a set of RGB images to Lab images.
        
        Arguments:
            data - set of RGB images

        Returns:
            set of Lab images corresponding to the inputted RGB images
        """
        return np.array([color.rgb2lab(rgb_img) for rgb_img in data])

