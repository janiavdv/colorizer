# Image Colorization with CNNs 

CSCI1430 Spring 2024 Final Project

We implemented a convolutional neural network (CNN) to colorize grayscale images using a U-Net architecture with the VGG-19 model. U-Net is a popular deep learning architecture known for its effectiveness in image segmentation tasks. VGG-19 is a large model with almost 150 million parameters that is pre-trained. It is traditionally used for feature detection and was adapted for colorizing in our project. Our model is trained using the MIT Places365 dataset, which contains 365,000 images of scenes (which we split into 328,500 train and 36,500 test images, a 90/10 split). Moreover, the model makes use of a custom Perceptual Loss function for a higher level chromatic evaluation of the CNN. Our results show that the model produces vibrant and realistically colored images. This project reinforces the potential of deep learning in creative image processing. Below is was our VGG-19 U-Net architecture.

![arch](https://github.com/johnsfarrell/rgbaddies/assets/69059806/87f86600-1eb1-4200-855d-d7c7c68c1260)

## Example Results

![grid](https://github.com/johnsfarrell/rgbaddies/assets/69059806/55e7740e-7baf-4fbd-b8e0-5b9568b22565)

## Usage

Download data:

```shell
cd data
./download.sh
```

Running the Flask server:

```shell
export FLASK_APP=api
flask run
```

Visualize images:

```shell
python3 visualize.py
```
