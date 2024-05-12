from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from skimage.io import imread
from util import load_weights
from skimage.transform import resize
import hyperparameters as hp

MODEL = load_weights("weights.h5")

def visualize_image(image_path):
    """
    Tests the model on image at path.

    Visualizes the original image (ground truth), grayscale image, and the predicted image.
    """
    img = imread(image_path)
    img = resize(img, output_shape=(hp.img_size, hp.img_size, 3))
    img / 255.0
    image_lab = rgb2lab(img)
    image_l = image_lab[:, :, [0, 0, 0]]

    # Run the model on image_l to get predicted ab channels
    image_ab = MODEL.predict(image_l[np.newaxis, ...])
    print(image_ab)
    image_ab = image_ab[0]
    image_lab[:, :, [1, 2]] = (image_ab * 1.5)
    image_rgb_predicted = lab2rgb(image_lab)
    print(image_rgb_predicted)
    # Create a single row plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    # Plot each image
    axs[0].imshow(image_l / 100.0)
    axs[0].set_title('Image Lightness Channel')
    axs[0].set_axis_off()

    axs[1].imshow(img)
    axs[1].set_title('Original RGB Image')
    axs[1].set_axis_off()

    axs[2].imshow(image_rgb_predicted)
    axs[2].set_title('Predicted RGB Image')
    axs[2].set_axis_off()

    mean_difference_across_channels = np.mean(abs(img - image_rgb_predicted), axis=2)
    axs[3].imshow(mean_difference_across_channels, cmap="RdYlGn_r")

    axs[3].set_title('Mean Differences')
    axs[3].set_axis_off()

    plt.tight_layout()
    plt.show()

    plt.imshow(image_lab[:, :, 1], cmap="gray")
    plt.show()
    plt.imshow(image_lab[:, :, 2], cmap="gray")
    plt.show()

"""
Usage: python visualize.py
"""
IMAGE_PATH = "test_images/test_image_1.jpg"
visualize_image(IMAGE_PATH)


def visualize_set_of_images(images_path, num_images):
    """
    Tests the model and shows off a bunch of images in grid.

    Visualizes the original image (ground truth), grayscale image, and the predicted image and difference.
    """

    fig, axs = plt.subplots(num_images, 4, figsize=(9, 9))

    for i in range(1, num_images + 1, 1):

        img = imread(images_path + str(i) + ".jpg")
        img = resize(img, output_shape=(hp.img_size, hp.img_size, 3))
        img / 255.0
        image_lab = rgb2lab(img)
        image_l = image_lab[:, :, [0, 0, 0]]

        # Run the model on image_l to get predicted ab channels
        image_ab = MODEL.predict(image_l[np.newaxis, ...])
        image_ab = image_ab[0]
        image_lab[:, :, [1, 2]] = (image_ab * 1.5)
        image_rgb_predicted = lab2rgb(image_lab)
        # Create a single row plot
        # Plot each image
        axs[i - 1][0].imshow(image_l / 100.0)
        axs[i - 1][0].set_axis_off()
        

        axs[i - 1][1].imshow(img)
        axs[i - 1][1].set_axis_off()

        axs[i - 1][2].imshow(image_rgb_predicted)
        axs[i - 1][2].set_axis_off()

        mean_difference_across_channels = np.mean(abs(img - image_rgb_predicted), axis=2)
        axs[i - 1][3].imshow(mean_difference_across_channels, cmap="RdYlGn_r")
        axs[i - 1][3].set_axis_off()

        if i < 2:
            axs[i - 1][0].set_title('Image Lightness Channel')
            axs[i - 1][1].set_title('Original RGB Image')
            axs[i - 1][2].set_title('Predicted RGB Image')
            axs[i - 1][3].set_title('Mean Differences')

    plt.tight_layout()
    plt.show()

num_ims = 5
# visualize_set_of_images("test_images/test_image_", num_ims)