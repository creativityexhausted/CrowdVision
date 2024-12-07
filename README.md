# CrowdVision
A deep learning-based crowd-counting solution using TensorFlow and Keras.
A Deep Learning-Based Crowd Counting Solution

# Overview:
CrowdVision is a machine learning project that uses convolutional neural networks (CNNs) to estimate crowd sizes in images. The solution handles diverse scenarios, including varying crowd densities, lighting conditions, and environments. This project is useful for public safety, event management, and crowd control.

# Features:
Preprocessing pipeline for image normalization and resizing.
CNN model architecture for accurate regression-based crowd size prediction.
Early stopping to prevent overfitting.
Prediction output is saved as a CSV file for analysis.
Adaptable for different datasets and crowd conditions.

# Dataset
1. Training Data: A folder containing labelled images (frames) and a corresponding label file (output.csv) with the number of people in each image.

2. Testing Data: A separate folder of images without labels (imageset.zip).

