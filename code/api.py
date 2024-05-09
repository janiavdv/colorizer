from flask import Flask, request, jsonify, send_file
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
import hyperparameters as hp
from model import Model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

def load_model(checkpoint="weights.h5"):
    """
    Load the model.

    Args:
        checkpoint (str): The path to the checkpoint file.

    Returns:
        Model: The model.
    """
    model = Model().mod
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.load_weights(checkpoint)
    return model

def resize_image(img, size=(256, 256)):
    """
    Resize the image to the specified size.

    Args:
        img (np.array): The image to resize.
        size (tuple): The target size.

    Returns:
        np.array: The resized image.
        tuple: The original size of the image.
    """
    return resize(img, size, anti_aliasing=True, mode='reflect')

def fix_dims(img):
    """
    Check if the image is grayscale or has an alpha channel.
    If it is grayscale, convert it to RGB.
    If it has an alpha channel, remove it.

    Args:
        img (np.array): The image to check.
    
    Returns:
        np.array: The image with the correct number of channels.
    """
    if len(img.shape) == 2: 
        img = gray2rgb(img)
    elif img.shape[2] == 4: 
        img = img[:,:,:3]
    return img

def color(img_rgb, output_lab=True):
    """
    Colorize the image.
    Convert the image to LAB color space, extract the L channel,
    predict the AB channels, and combine them to form the colorized image.

    Args:
        img_rgb (np.array): The image to colorize (RGB).
        output_lab? (bool): Whether to output the image in LAB or RGB.

    Returns:
        np.array: The colorized image in LAB (default) or RGB.
    """
    image_lab = rgb2lab(img_rgb)
    l = image_lab[:, :, [0, 0, 0]]
    predicted_ab = MODEL.predict(l[np.newaxis, ...])[0]
    image_lab[:, :, [1, 2]] = predicted_ab
    return image_lab if output_lab else lab2rgb(image_lab * 1)

def upscale_color(original_img, color_lab):
    """
    Upscale the colorized image to the original size.

    Args:
        original_img (np.array): The original image in RGB.
        color_lab (np.array): The colorized image in LAB.

    Returns:
        np.array: The upscaled colorized image in RGB.
    """
    shape = original_img.shape[:2]
    color_lab = resize_image(color_lab, shape)
    original_lab = rgb2lab(original_img)

    ab = color_lab[:, :, 1:]
    original_lab[:, :, 1:] = ab

    return lab2rgb(original_lab)


def rgb_to_byte_arr(img_rgb):
    """
    Convert the RGB image to a byte array.
    PIL is used to convert the image to a byte array.
    Purpose is to send the image as a response.

    Args:
        img_rgb (np.array): The RGB image.
    
    Returns: 
        io.BytesIO: The byte array.
    """
    img_rgb = (img_rgb * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_rgb)
    byte_arr = io.BytesIO()
    img_pil.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr

MODEL = load_model()

@app.route('/api', methods=['POST'])
def api():
    """
    The API endpoint.
    The image is received as a POST request.
    The image is resized, checked for the correct number of channels,
    colorized, and sent back as a response.

    Development server:
        1. `export FLASK_APP=api`
        2. `flask run`

    Returns:
        Response: The colorized image.

    Raises:
        JSONDecodeError: If the request does not contain an image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    img = imread(request.files['file'])
    img = fix_dims(img)

    downscaled_img = resize_image(img, (hp.img_size, hp.img_size))
    colored_lab = color(downscaled_img)
    upscaled_rgb = upscale_color(img, colored_lab)

    return send_file(rgb_to_byte_arr(upscaled_rgb), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
