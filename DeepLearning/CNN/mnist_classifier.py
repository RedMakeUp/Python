import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.metrics import accuracy_score

# Plot some details about the dataset and show some example points
def showDatasetExamples(xTrain, yTrain, xTest, yTest):
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('MINIST Dataset Examples')
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 6])
    # Subplot "Summary"
    ax_summary = plt.subplot(gs[0])
    ax_summary.set_xticks([])
    ax_summary.set_yticks([])
    ax_summary.set_title('Dataset Summary', fontsize=20, fontweight='bold')
    ax_summary.axis('off')
    ax_summary.axhline(1.0, color='black')
    ax_summary_text_size = 12
    ax_summary_mono = {'family' : 'monospace'}
    ax_summary.text(0.14, 0.6, "Each image size:         28*28*1", fontsize=ax_summary_text_size, fontdict=ax_summary_mono)
    ax_summary.text(0.14, 0.3, "Train set image numbers: {}".format(xTrain.shape[0]), fontsize=ax_summary_text_size, fontdict=ax_summary_mono)
    ax_summary.text(0.14, 0.0, "Test set image numbers:  {}".format(xTest.shape[0]), fontsize=ax_summary_text_size, fontdict=ax_summary_mono)
    # Subplot "Examples"
    ax_examples = plt.subplot(gs[2])
    ax_examples.set_xticks([])
    ax_examples.set_yticks([])
    ax_examples.set_title('Dataset Examples', fontsize=20, fontweight='bold')
    ax_examples.axis('off')
    ax_examples.axhline(1.0, color='black')
    ax_examples_inners = gridspec.GridSpecFromSubplotSpec(3, 5, gs[2], wspace=0.1, hspace=0.1)
    for i in range(ax_examples_inners.nrows):
        for j in range(ax_examples_inners.ncols):
            ax = fig.add_subplot(ax_examples_inners[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            index = i * ax_examples_inners.nrows + j
            ax.imshow(xTrain[index], cmap='binary', interpolation='nearest')
            ax.text(0.05, 0.05, str(yTrain[index]), transform=ax.transAxes, color='green')

    plt.show()

# Define model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Train a batch
# @iamges shape with (batch, width. height, channels)
# @labels shape with (labels)
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# Download MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Show examples
#showDatasetExamples(x_train, y_train, x_test, y_test)

# Prepare the data
x_train = x_train / 255# Normalize
x_test = x_test / 255
x_train = x_train[..., tf.newaxis]# (60000, 28, 28, ) to (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)# Contruct "Dataset" structure using the data
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

EPOCHS = 5
history = {
    'loss': np.zeros(EPOCHS),
    'accuracy': np.zeros(EPOCHS),
    'val_loss': np.zeros(EPOCHS),
    'val_accuracy': np.zeros(EPOCHS)
}

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        # tf.config.experimental_run_functions_eagerly(True)
        train_step(images, labels)
        # tf.config.experimental_run_functions_eagerly(False)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result() * 100,
        test_loss.result(),
        test_accuracy.result() * 100
    ))

    history['loss'][epoch] = train_loss.result()
    history['accuracy'][epoch] = train_accuracy.result()
    history['val_loss'][epoch] = test_loss.result()
    history['val_accuracy'][epoch] = test_accuracy.result()


# Test
model.summary()
 
# for i in range(10):
#     print(str(y_test[i]))
#     inputs = x_test[i]
#     inputs = inputs[tf.newaxis, ...]
#     prediction = model(inputs, training=False)
#     print(np.argmax(prediction))

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

