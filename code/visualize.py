from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import numpy as np
from skimage.io import imread
from api import load_model

MODEL = load_model()

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
    image_ab = MODEL.predict(image_l[np.newaxis, ...])
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

    plt.imshow(image_ab[:, :, 0], cmap="gray")
    plt.show()

def clear(plt):
    """
    Clear the plot.
    """
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


"""
Usage: python visualize.py
"""
IMAGE_PATH = "test_images/test_image_1.jpg"
test(IMAGE_PATH)
