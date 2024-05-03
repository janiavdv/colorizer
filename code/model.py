import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, Rescaling, Reshape, Dropout, Flatten, Dense, BatchNormalization
from keras.applications import ResNet50V2

import hyperparameters as hp

IMG_SIZE = hp.img_size

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.resnet = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(hp.img_size, hp.img_size, 3),
        )
        for layer in self.resnet.layers:
            layer.trainable = False


        self.head = [ 
            Conv2DTranspose(256, 3, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2DTranspose(128, 3, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2DTranspose(64, 3, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2DTranspose(32, 3, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2DTranspose(16, 3, 2, activation="relu", padding="same"),
            BatchNormalization(),
            Conv2DTranspose(2, 3, activation="sigmoid", padding="same"),
            Rescaling(scale=255.0, offset=-128)
        ]

        # Don't change the below:
        self.resnet = tf.keras.Sequential(self.resnet, name="resnet_base")
        self.head = tf.keras.Sequential(self.head, name="resnet_head")
        
    def call(self, x):
        """ Passes input image through the network. """
        x = self.resnet(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
       """ Loss function for the model. """
       return tf.keras.losses.MSE(labels, predictions)
        