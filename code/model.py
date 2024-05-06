import keras
from keras.layers import \
     concatenate, Conv2DTranspose, Rescaling, BatchNormalization
from keras.applications import VGG19
import hyperparameters as hp

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

        # block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   
        # block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808 
        # block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080 
        # block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584       
        # block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928   

        self.mod = vgg19.output

        b = Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation="relu", padding="same")(self.mod)
        b = BatchNormalization()(b)
        self.mod = concatenate([b, vgg19.get_layer("block5_conv4").output])

        block_layer_sizes = [
            (512, "block4_conv4"),
            (256, "block3_conv4"),
            (128, "block2_conv2"),
            (64, "block1_conv2")
        ]

        for filters, layer_name in block_layer_sizes:
            b = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, activation="relu", padding="same")(self.mod)
            b = BatchNormalization()(b)
            b = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, activation="relu", padding="same")(b)
            b = BatchNormalization()(b)
           
            self.mod = concatenate([b, vgg19.get_layer(layer_name).output])
        
        
        # Final resizing.
        self.mod = Conv2DTranspose(64, 3, activation="relu", padding="same")(self.mod)
        self.mod = Conv2DTranspose(2, 3, activation="sigmoid", padding="same")(self.mod)
        self.mod = Rescaling(scale=255.0, offset=-128)(self.mod)
        self.mod = keras.Model(inputs=inp, outputs=self.mod)
   