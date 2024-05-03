from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
import numpy as np
from PIL import Image
from tqdm import tqdm
import hyperparameters as hp
from model import Model
import tensorflow as tf
import os
from skimage.io import imread

checkpoint = "weights.e049-acc170.1781.h5"
model = Model()
model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
model.load_weights(checkpoint)



def test(image_path, out_dir="output"):
    """
    Tests the model on image at path.

    Visualizes the original image (ground truth), grayscale image, and the predicted image.
    """
    img = imread(image_path)
    # if image is graycale, convert to RGB
    if len(img.shape) == 2:
        image_rgb = gray2rgb(img)

    image_lab = rgb2lab(img)
    image_l = image_lab[:, :, [0, 0, 0]]
    image_a_original = image_lab[:, :, [1]]
    image_b_original = image_lab[:, :, [2]]
    # Run the model on image_l to get predicted ab channels
    image_ab = model.predict(image_l[np.newaxis, ...])
    print(image_ab)
    image_ab = image_ab[0]
    image_lab[:, :, [1, 2]] = image_ab
    image_rgb_predicted = lab2rgb(image_lab * 1.3)
    print(image_rgb_predicted)
    # Create a single row plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each image
    axs[0].imshow(image_l / 100.0)
    axs[0].set_title('Image L')

    axs[1].imshow(img)
    axs[1].set_title('Original Image')

    axs[2].imshow(image_rgb_predicted)
    axs[2].set_title('Predicted RGB Image')

    plt.tight_layout()
    plt.show()



def clear(plt):
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


"""
Usage: python visualize.py
"""
# visualize_images = range(1, 36501)
# visualize_images = np.random.choice(visualize_images, 50, replace=False)
test("test_imag.jpg")
