import keras
from keras.layers import \
     concatenate, Conv2DTranspose, Rescaling, BatchNormalization
from keras.applications import VGG19
import hyperparameters as hp
import numpy as np
from skimage.filters import gaussian
from keras.losses import mean_squared_error as mse

class Model():
    def __init__(self):
        
        inp = keras.Input(shape=(hp.img_size, hp.img_size, 3))
        
        vgg19 = VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(hp.img_size, hp.img_size, 3),
            input_tensor = inp
        )

        for layer in vgg19.layers:
            layer.trainable = False

        """
        Model: "vgg19"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                        
        block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                        
        block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                        
        block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                        
        block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                        
        block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                        
        block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                        
        block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                        
        block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                        
        block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                        
        block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                        
        block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                        
        block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                        
        block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                        
        block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                        
        block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                        
        block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                        
        block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                        
        block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                        
        block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                        
        block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                        
        block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                        
        flatten (Flatten)           (None, 25088)             0         
                                                                        
        fc1 (Dense)                 (None, 4096)              102764544 
                                                                        
        fc2 (Dense)                 (None, 4096)              16781312  
                                                                        
        predictions (Dense)         (None, 1000)              4097000   
                                                                        
        =================================================================
        """
        self.mod = vgg19.output

        block_layer_sizes = [
            ["block5_conv4", (512, 1), (512, 2), (512, 2), (512, 2)],
            ["block4_conv4", (512, 1), (512, 1), (512, 1), (512, 2)],
            ["block3_conv4", (256, 1), (256, 1), (256, 1), (256, 2)],
            ["block2_conv2", (128, 1), (128, 2)],
            ["block1_conv2", (64, 1), (64, 2)]
        ]

        for block in block_layer_sizes:
            block_name = block[0]
            b = Conv2DTranspose(filters=block[1][0], kernel_size=3, strides=block[1][1], activation="relu", padding="same")(self.mod)
            b = BatchNormalization()(b)
            for i in range(2, len(block)):
                b = Conv2DTranspose(filters=block[i][0], kernel_size=3, strides=block[i][1], activation="relu", padding="same")(b)
                b = BatchNormalization()(b)
            
            self.mod = concatenate([b, vgg19.get_layer(block_name).output])

        self.mod = Conv2DTranspose(2, 3, activation="sigmoid", padding="same")(self.mod)
        self.mod = Rescaling(scale=255.0, offset=-128)(self.mod)
        self.mod = keras.Model(inputs=inp, outputs=self.mod)

    def blur(self, img, kernel_size):
        if type(img) != np.ndarray:
            img = img.numpy()
        return gaussian(img, sigma=kernel_size)
   
    def percept_loss_func(self, truth, predicted):
        truth_blur_3 = self.blur(truth, 3)
        truth_blur_5 = self.blur(truth, 5)

        predicted_blur_3 = self.blur(predicted, 3)
        predicted_blur_5 = self.blur(predicted, 5)

        dist = mse(truth, predicted)
        dist_3 = mse(truth_blur_3, predicted_blur_3)
        dist_5 = mse(truth_blur_5, predicted_blur_5)
        
        return np.sum([dist, dist_3, dist_5]) / 3
