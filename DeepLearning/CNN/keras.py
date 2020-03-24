from matplotlib import pyplot
from tensorflow import keras
import numpy as np
import sys

# Load train and test dataset
def load_dataset():
    # Load dataset
    (trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()
    # One hot encode target values
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

# Scale pixels
def prep_pixels(train, test):
    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalize to range [0, 1]
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # Return normalized images
    return train_norm, test_norm

# Define CNN model
def define_model():
    model = keras.Sequential()
    # ...
    return model

# Plot diagnostic learning curves
def sumarize_disgnostices(history):
    # Plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_loss'], color = 'orange', label = 'validation')
    # Plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color = 'blue', label = 'train')
    pyplot.plot(history.history['vak_accuracy'], color = 'orange', label = 'validation')
    # Save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# Run the test harness for evaluating a model
def run_test_harness():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # Prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # Define model
    model = define_model()
    history = model.fit(trainX, trainY, epochs = 100, batch_size = 64, validation_data = (testX, testY), verbose = 0)
    # Evaludate model
    _, acc = model.evaluate(testX, testY, verbose = 0)
    print('> %.3f' & (acc * 100.0))
    # Learning curves
    sumarize_disgnostices(history)