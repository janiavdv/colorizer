"""
Number of epochs to train the model for.
"""
num_epochs = 50

"""
Learning rate for the optimizer.
"""
learning_rate = 1e-2

"""
Resize image size.
"""
img_size = 224

"""
Maximum number of weight files to save to checkpoint directory. If
set to a number <= 0, then all weight files of every epoch will be
saved. Otherwise, only the weights with highest accuracy will be saved.
"""
max_num_weights = 10

"""
Defines the number of training examples per batch.
"""
batch_size = 64

"""
Number of steps per epoch.
"""
train_length = 328500
test_length = 36500
steps_per_epoch = (train_length // batch_size) // 32
validation_steps = (test_length // batch_size) // 32
