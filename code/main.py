"""
Originally for
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University

Edited by Jania Vandevoorde for final project. 
"""

import os
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from model import Model
from preprocess import Datasets
from tensorboard_utils import CustomModelSaver

from matplotlib import pyplot as plt
import keras

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's colorize some images!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.'
    )
    
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.'''
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.'''
    )

    return parser.parse_args()

def train(model, datasets, checkpoint_path="checkpoints/", logs_path="logs/", init_epoch=0):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        initial_epoch=init_epoch,
        callbacks=callback_list
    ) # TODO: do we need to specify batch size here?


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    datasets = Datasets(ARGS.data)

    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
        "model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "model" + \
        os.sep + timestamp + os.sep

    # Print summary of model
    # model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=[keras.metrics.MeanSquaredError()]
    )

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)

# Make arguments global
ARGS = parse_args()

if __name__ == "__main__":
    main()