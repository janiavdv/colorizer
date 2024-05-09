import keras
from keras.layers import \
     concatenate, Conv2DTranspose, Rescaling, BatchNormalization
from keras.applications import VGG19
from keras.layers import LeakyReLU
import hyperparameters as hp
import numpy as np
from keras.losses import mean_squared_error as mse
from skimage.filters import gaussian
import tensorflow as tf

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
        b = Conv2DTranspose(filters=512, kernel_size=3, strides=2, activation=None, padding="same")(self.mod)
        b = LeakyReLU(alpha=0.1)(b)
        b = BatchNormalization()(b)
        self.mod = concatenate([b, vgg19.get_layer("block5_conv4").output])

        block_layer_sizes = [
            (512, "block4_conv4"),
            (256, "block3_conv4"),
            (128, "block2_conv2"),
            (64, "block1_conv2")
        ]

        for filters, layer_name in block_layer_sizes:
            b = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, activation=None, padding="same")(self.mod)
            b = LeakyReLU(alpha=0.1)(b)
            b = BatchNormalization()(b)
            b = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, activation=None, padding="same")(b)
            b = LeakyReLU(alpha=0.1)(b)
            b = BatchNormalization()(b)
           
            self.mod = concatenate([b, vgg19.get_layer(layer_name).output])

        self.mod = Conv2DTranspose(64, 3, activation=None, padding="same")(self.mod)
        self.mod = LeakyReLU(alpha=0.1)(self.mod)
        self.mod = BatchNormalization()(self.mod)
        self.mod = Conv2DTranspose(2, 3, activation="sigmoid", padding="same")(self.mod)
        self.mod = Rescaling(scale=255.0, offset=-128)(self.mod)
        self.mod = keras.Model(inputs=inp, outputs=self.mod)

    def get_gaussian_kernel(self, shape=(3,3), sigma=0.5):
        """build the gaussain filter"""
        m,n = [(ss-1.)/2. for ss in shape]
        x = tf.expand_dims(tf.range(-n,n+1,dtype=tf.float32),1)
        y = tf.expand_dims(tf.range(-m,m+1,dtype=tf.float32),0)
        h = tf.exp(tf.math.divide_no_nan(-((x*x) + (y*y)), 2*sigma*sigma))
        h = tf.math.divide_no_nan(h,tf.reduce_sum(h))
        return h

    def gaussian_blur(self, inp, shape=(3,3), sigma=0.5):
        """Convolve using tf.nn.depthwise_conv2d"""
        in_channel = tf.shape(inp)[-1]
        k = self.get_gaussian_kernel(shape,sigma)
        k = tf.expand_dims(k,axis=-1)
        k = tf.repeat(k,in_channel,axis=-1)
        k = tf.reshape(k, (*shape, in_channel, 1))
        # using padding same to preserve size (H,W) of the input
        conv = tf.nn.depthwise_conv2d(inp, k, strides=[1,1,1,1],padding="SAME")
        return conv
   
    def percept_loss_func(self, truth, predicted):
        truth_blur_3 = self.gaussian_blur(truth)
        truth_blur_5 = self.gaussian_blur(truth)

        predicted_blur_3 = self.gaussian_blur(predicted)
        predicted_blur_5 = self.gaussian_blur(predicted)

        dist = mse(truth, predicted) ** 0.5
        dist_3 = mse(truth_blur_3, predicted_blur_3) ** 0.5
        dist_5 = mse(truth_blur_5, predicted_blur_5) ** 0.5
        
        return np.sum([dist, dist_3, dist_5]) / 3
