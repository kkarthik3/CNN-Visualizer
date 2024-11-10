import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
import math

MODEL_PATH = 'cnn_mnist_model.h5'

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set page layout and title
st.title("CNN Number Classification")

st.info("This app uses a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset. "
        "The CNN model can visualize filters, feature maps, and outputs from different layers.")

# Get user input for selecting a number
st.info("Select a digit (0-9) to view a sample image and analyze the CNN's processing of this image.")
number = st.number_input('Enter a number to select a sample image (0-9):', min_value=0, max_value=9, value=0)

# Select a random image corresponding to the user-input number
indices = np.where(y_train == number)[0]
selected_index = np.random.choice(indices)
selected_image = x_train[selected_index]

st.write(f"### Selected Sample Image for number {number}")
st.image(selected_image, caption=f'Sample Image of number {number}', use_column_width=True)

# Build the CNN model
def create_model():
    st.info("Building the Convolutional Neural Network (CNN) model with layers that include convolutional, pooling, and dense layers.")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Check if the model exists, if not train and save the model
if os.path.exists(MODEL_PATH):
    st.info("Loading the pretrained model from file to save time on training.")
    model = tf.keras.models.load_model(MODEL_PATH)
    st.write("Model loaded from file.")
else:
    st.info("Training the CNN model on the MNIST dataset for 5 epochs and saving it for future use.")
    model = create_model()
    history = model.fit(x_train[..., tf.newaxis], y_train, epochs=5, validation_split=0.2)
    model.save(MODEL_PATH)
    st.write("Model trained and saved to file.")
    
    # Plot training history
    st.write("### Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='accuracy')
    ax.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

# Display model architecture
st.write("### Model Architecture")
st.info("Showing the structure of the CNN model to understand the arrangement of layers.")
model.summary(print_fn=lambda x: st.text(x))

# Function to visualize the filters
def plot_filters(layer, x, y):
    st.info("Visualizing the filters (or kernels) in the first convolutional layer. These filters help the model detect edges, textures, and other patterns in the image.")
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters, ix = filters.shape[-1], 1
    fig, ax = plt.subplots(x, y, figsize=(8, 8))
    for i in range(n_filters):
        f = filters[:, :, :, i]
        ax = plt.subplot(x, y, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, 0], cmap='gray')
        ix += 1
    st.pyplot(fig)

# Visualize filters of the first conv layer
st.write("### Filters of the First Conv Layer")
plot_filters(model.layers[0], 4, 8)

# Function to visualize feature maps
def plot_feature_maps(model, layer_index, input_image):
    st.info("Generating feature maps from the first convolutional layer. Feature maps show how the input image is transformed by the model to detect patterns.")
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    feature_maps = intermediate_model.predict(input_image)
    
    num_features = feature_maps.shape[-1]
    grid_size = math.ceil(math.sqrt(num_features))
    
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(num_features):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    st.pyplot(fig)

# Visualize feature maps of the first conv layer
st.write("### Feature Maps of the First Conv Layer")
plot_feature_maps(model, 0, selected_image[tf.newaxis, ..., tf.newaxis])

# Function to visualize dense layer output
def plot_dense_output(model, layer_index, input_image):
    st.info("Visualizing the dense layer output, which summarizes the learned features before classification.")
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    dense_output = intermediate_model.predict(input_image)
    
    st.write("### Dense Layer Output")
    st.write(dense_output)

# Visualize dense layer output
plot_dense_output(model, -2, selected_image[tf.newaxis, ..., tf.newaxis])

# Function to visualize pooling layer output
def plot_pooling_output(model, layer_index, input_image):
    st.info("Visualizing the pooling layer output, which reduces dimensionality while preserving essential features.")
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    pooling_output = intermediate_model.predict(input_image)
    
    num_features = pooling_output.shape[-1]
    grid_size = math.ceil(math.sqrt(num_features))
    
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(num_features):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(pooling_output[0, :, :, i], cmap='gray')
    st.pyplot(fig)

# Visualize pooling layer output
st.write("### Pooling Layer Output")
plot_pooling_output(model, 1, selected_image[tf.newaxis, ..., tf.newaxis])

# Function to visualize convolutional layer output
def plot_conv_output(model, layer_index, input_image):
    st.info(f"Visualizing the output of the convolutional layer at index {layer_index}. Convolutional layers transform the image into feature-rich outputs.")
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    conv_output = intermediate_model.predict(input_image)
    
    num_features = conv_output.shape[-1]
    grid_size = math.ceil(math.sqrt(num_features))
    
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(num_features):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(conv_output[0, :, :, i], cmap='gray')
    st.pyplot(fig)

# Visualize convolutional layer output
st.write("### Convolutional Layer Output")
plot_conv_output(model, 0, selected_image[tf.newaxis, ..., tf.newaxis])
plot_conv_output(model, 2, selected_image[tf.newaxis, ..., tf.newaxis])
