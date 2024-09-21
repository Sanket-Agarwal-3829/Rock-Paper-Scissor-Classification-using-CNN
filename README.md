# Rock-Paper-Scissors Classifier

## Overview

This repository contains a Convolutional Neural Network (CNN) model designed to classify images of hand gestures representing Rock, Paper, and Scissors. The model uses TensorFlow and Keras to build and train the classifier based on a dataset of labeled images for each hand gesture. It can predict the gesture from an image input, making it suitable for Rock-Paper-Scissors games or hand gesture recognition applications.

## Model Architecture

The model is a sequential CNN with the following layers:

1. Conv2D Layer (32 filters, 3x3 kernel size) - Activation: `ReLU`  
   The first convolution layer extracts features from the input image with 32 filters and a 3x3 kernel.
   
2. MaxPooling2D (2x2 pool size)
   Reduces the dimensionality by selecting the maximum value from each 2x2 block, retaining important information while reducing computation.
   
3. Conv2D Layer (64 filters, 3x3 kernel size) - Activation: `ReLU`  
   A deeper layer with 64 filters to capture more complex patterns.
   
4. MaxPooling2D (2x2 pool size) 
   Further reduces the dimensionality.
   
5. Conv2D Layer (64 filters, 3x3 kernel size) - Activation: `ReLU`  
   Another convolution layer to further refine feature extraction.
   
6. MaxPooling2D (2x2 pool size) 
   Further reduces the size of the feature map.

7. Conv2D Layer (128 filters, 3x3 kernel size) - Activation: `ReLU`  
   This layer further captures intricate patterns in the image data.
   
8. MaxPooling2D (2x2 pool size)
   Final pooling layer to reduce dimensionality before flattening.
   
9. Flatten Layer  
   Converts the 2D feature maps into a 1D feature vector for the fully connected layers.
   
10. Dropout Layer (0.4) 
    Regularization to prevent overfitting by randomly setting 40% of the neurons to zero during training.
   
11. Dense Layer (512 units) - Activation: `ReLU`  
    Fully connected layer with 512 units for complex decision-making.

12. Output Dense Layer (3 units) - Activation: `Softmax`  
    Output layer with 3 units (one for each class: Rock, Paper, Scissors) using softmax activation for multi-class classification.

## Compilation

The model is compiled with the following configurations:

- Optimizer: `Adam` - Adaptive learning rate optimization.
- Loss Function: `Categorical Crossentropy` - Suitable for multi-class classification.
- Metrics: `Accuracy` - To monitor the accuracy during training.
