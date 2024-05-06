import os
import numpy as np
from PIL import Image
import hyperparameters as hp
from skimage import color
import keras

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path    

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.x_train, self.y_train = self.get_data(os.path.join(self.data_path, "train/"), True)
        self.x_test, self.y_test = self.get_data(os.path.join(self.data_path, "test/"), False)   
             
    def get_data(self, path, shuffle):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"

        Returns:
            An iterable image-batch generator
        """
        file_paths = []
        for root, _, files in os.walk(path):
            for name in files:
                if name.endswith(".jpeg"):
                    file_paths.append(os.path.join(root, name)) 
        x = []
        y = []
        for i, file_path in enumerate(file_paths):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            img = img[:, :, :3]
            l, ab = self.rgb_to_lab(img)
            x.append(l)
            y.append(ab)

        x = np.array(x)
        y = np.array(y)
        x = np.expand_dims(x, axis=-1)
        return (x, y)

    def rgb_to_lab(self, rgb_img):
        """ Converts a RGB image to Lab image

        Arguments:
            rgb_img - rgb image matrix
        
        Returns:
            lab image, as tuple (l, ab)
        """
        lab_img = color.rgb2lab(rgb_img)
        l = lab_img[:, :, [0, 0, 0]]
        ab = lab_img[:, :, [1, 2]]

        # l = lab_img[:, :, 0]
        # ab = lab_img[:, :, 1:]

        return (l, ab)
    
class JDatasets:
    """
    Class for containing the training and test sets, as well as other data related functions.
    """

    def __init__(self, data_path):
        """
        Initialize the Dataset object.
        """
        self.data_path = data_path

        self.train_data = self.get_data(os.path.join(self.data_path, "train/"))

        self.test_data = self.get_data(os.path.join(self.data_path, "test/"))

    def get_data(self, path, shuffle=False, augment=True):
        """
        Gets the data at path, shuffling and augmenting as desired.
        """
        if augment:
            data = keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                horizontal_flip=True,
            )
        else:
            data = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self.preprocess_fn)

        img_size = hp.img_size

        data = data.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=hp.batch_size,
            shuffle=shuffle,
        )

        return self.data_rgb_to_l_ab(data)

    def data_rgb_to_l_ab(self, data):
        """
        Converts the RGB data to L+AB data.
        """
        for im in data:
            im = im[0]
            im_lab = color.rgb2lab(im)
            # We need 3 identical channels because of our pretrained model backbones.
            im_l = im_lab[:, :, :, [0, 0, 0]]
            im_ab = im_lab[:, :, :, [1, 2]]
            yield (im_l, im_ab)

    def preprocess_fn(self, img):
        """Preprocess function for ImageDataGenerator."""
        img = img / 255.0
        return img
