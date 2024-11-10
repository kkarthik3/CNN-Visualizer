# 🧠 CNN MNIST Digit Classification Visualizer

## 📝 Overview
This Streamlit application provides an interactive visualization of how a Convolutional Neural Network (CNN) processes and classifies handwritten digits from the MNIST dataset. The app allows users to explore various aspects of the CNN, including filters, feature maps, and layer outputs.

## ⭐ Features
- 🔢 Interactive digit selection (0-9) with random sample image display
- 🏗️ Visualization of CNN model architecture
- 📊 Real-time display of:
  - 🔍 Convolutional layer filters
  - 🗺️ Feature maps
  - 📉 Pooling layer outputs
  - 📈 Dense layer outputs
- 📚 Model training history visualization (for newly trained models)
- 💾 Automatic model saving and loading functionality

## 🛠️ Requirements
```
tensorflow
streamlit
matplotlib
numpy
```

## 🚀 Installation
1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage
1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Use the number input selector to choose a digit (0-9) and explore different visualizations

## ⚙️ How It Works

### 🏗️ Model Architecture
The CNN model consists of:
- 2 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- 2 Dense layers (including softmax output layer)

### 🔍 Visualization Components
1. **Filters Visualization**: Displays the learned filters from the first convolutional layer
2. **Feature Maps**: Shows how the input image is transformed by convolutional layers
3. **Pooling Layer Output**: Demonstrates the dimensionality reduction process
4. **Dense Layer Output**: Visualizes the final features before classification

## 🎯 Model Training
- The model is trained on the MNIST dataset
- Training occurs only if no pre-trained model is found
- Model is automatically saved after training for future use
- Training history is displayed with accuracy metrics
