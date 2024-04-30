import tensorflow as tf
from keras.layers import \
    Conv2D, Conv2DTranspose, Rescaling, Dropout, Flatten, Dense, BatchNormalization

import hyperparameters as hp

IMG_SIZE = hp.img_size

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        
        self.architecture = [
            # output: 112 x 112 x 64
            Conv2D(64, (3,3), activation='relu', strides=(2,2),padding='same'), 
            Conv2D(64, (3,3), activation='relu', strides=(2,2),padding='same'), 
            BatchNormalization(),

            # output: 56 x 56 x 128
            Conv2D(128, (3,3), activation='relu', strides=(2,2),padding='same'), 
            Conv2D(128, (3,3), activation='relu', strides=(2,2),padding='same'), 
            BatchNormalization(),
            
            # output: 28 x 28 x 256
            Conv2D(256, (3,3), activation='relu', strides=(2,2),padding='same'), 
            Conv2D(256, (3,3), activation='relu', strides=(2,2),padding='same'), 
            BatchNormalization(),

            # output: 28 x 28 x 512
            Conv2D(512, (3,3), activation='relu', strides=(2,2),padding='same'), 
            Conv2D(512, (3,3), activation='relu', strides=(2,2),padding='same'), 
            BatchNormalization(),

            # output: 28 x 28 x 512
            Conv2D(512, (3,3), activation='relu', strides=(1,1),padding='same'), 
            Conv2D(512, (3,3), activation='relu', strides=(1,1),padding='same'), 
            BatchNormalization(),


            # output: 28 x 28 x 512
            Conv2D(512, (3,3), activation='relu', strides=(1,1),padding='same'), 
            Conv2D(512, (3,3), activation='relu', strides=(1,1),padding='same'), 
            BatchNormalization(),

            # output: 56 x 56 x 256
            Conv2D(256, (3,3), activation='relu', strides=(1,1),padding='same'), 
            Conv2D(256, (3,3), activation='relu', strides=(1,1),padding='same'), 
            BatchNormalization(),

            Conv2DTranspose(2, 3, strides=1, activation="relu", padding="same"),
            Rescaling(scale=255.0, offset=-128),
        ] 
        
        
    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
       """ Loss function for the model. """
       return tf.keras.losses.MSE(labels, predictions)
        