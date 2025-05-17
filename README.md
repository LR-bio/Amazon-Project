Amazon Earthworks Detector

Overview:

This project uses artificial intelligence (AI) to find ancient earthworks in 
the Amazon rainforest. Earthworks are human-made shapes like mounds, patterns, 
or raised fields that were built by people long ago. The project works by looking 
at elevation data, which shows the shape of the land, to spot these features.

The area focused on is the Guaporé River Basin in Brazil. Recent 
technology like LiDAR and satellite images have discovered many of 
these hidden earthworks there. This project helps archaeologists by 
automatically detecting these features from the elevation data.

The system works in a few steps: first, it creates fake (synthetic) 
elevation maps with and without earthworks to train the AI. Then, 
it processes these maps to highlight important details like slopes 
and smooths out the data to reduce noise. After that, it trains a 
computer model called a convolutional neural network (CNN) to tell 
if an earthwork is present or not. Finally, it can analyze new elevation 
maps and say whether they likely contain earthworks.

The project includes code to generate training data, train the model, 
and test it on new elevation tiles. You can use it by running simple 
commands in the terminal. It uses Python programming and some common 
libraries like PyTorch, NumPy, and Rasterio.

In the future, this project could be improved by using real elevation 
data collected from satellites and LiDAR scanners. It could also be 
expanded to locate exactly where earthworks are on the map instead of 
just saying if they exist or not. There’s also potential to build a 
user-friendly app to make it easier for researchers to use.



Requirements
  -Python 3.8 or higher
  -TensorFlow 2.x
  -NumPy
  -OpenCV (cv2)
  -Rasterio
  -Matplotlib

Installation of required packages can be done via pip:
  pip install tensorflow numpy opencv-python rasterio matplotlib

Dataset Preparation
  Input data should consist of georeferenced satellite or elevation images in TIFF format.   These images will be processed into 128x128 or 64x64 pixel tiles suitable for CNN input. Data augmentation techniques such as flipping and rotation are applied to improve model robustness.

Organize your dataset directory with the following structure:
dataset/
  images/
    image1.tif
    image2.tif
    ...
  labels/
    image1_label.npy
    image2_label.npy
    ...
Ensure label files correspond accurately to their respective image tiles.

Usage
  -Prepare your dataset by converting satellite images into tiles and generating corresponding labels. Use the provided load_and_preprocess_data() function as a template for this step.
  -Train the model by running the script. The model uses binary cross-entropy loss and the Adam optimizer. Training parameters such as epochs and batch size can be adjusted in the script.
  -Evaluate and save the model. The trained model will be saved as amazon_archaeology_cnn_model.h5.
  -Predict on new images: Use the predict_tile() function to classify new image tiles for archaeological features.

Example usage in Python:
  # Load data
    X, Y = load_and_preprocess_data('dataset/images', 'dataset/labels')
  # Train model
    model = build_model()
    train_model(model, X, Y)
  # Save model
    model.save('amazon_archaeology_cnn_model.h5')
  # Predict on new data
    image_tile = ...  # Load new tile as numpy array
    prediction = predict_tile(model, image_tile)
    print(f"Prediction: {'Feature present' if prediction > 0.5 else 'No feature detected'}")

Model Architecture
  The CNN consists of:
    -Multiple convolutional layers with ReLU activation and max pooling
    -Flatten layer feeding into dense layers
    -Output layer with sigmoid activation for binary classification
    -This architecture balances performance and computational efficiency, suitable for training on moderate GPU resources.

Limitations and Considerations:
  -The model requires georeferenced TIFF images for best results.
  -High-quality labeled data is critical for successful training.
  -Processing large satellite images may require significant storage and memory.
  -The model currently performs binary classification and does not identify feature types.
  -Further tuning and validation with diverse datasets is recommended for deployment.

Future Work:
  -Expand the model to multi-class classification for different feature types.
  -Integrate LiDAR data and multi-spectral imagery.
  -Develop a complete pipeline for geospatial data preprocessing and post-processing visualization.
  -Explore transfer learning to improve accuracy and reduce training time.

License
This project is provided under the MIT License.
