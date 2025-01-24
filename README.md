# Disease Classification Model

**Project Overview**

This project focuses on developing a robust disease classification model using an ensemble of deep learning architectures. By leveraging the strengths of multiple pre-trained models, the system achieves improved accuracy and reliability in classifying medical images into disease categories. The model is designed to aid healthcare professionals in diagnosing diseases effectively using image-based analysis.

# Features

Utilizes ensemble learning with ResNet101V2 and VGG19 architectures.

Incorporates data augmentation techniques to improve model generalization.

Implements callbacks like EarlyStopping and ReduceLROnPlateau for optimized training.

Evaluates model performance using metrics such as accuracy, precision, recall, and F1-score.

# Implementation Details

**1. Data Preprocessing**

The dataset is preprocessed using the ImageDataGenerator class from TensorFlow. This includes:

Rescaling pixel values to normalize image data.

Applying data augmentation techniques like rotation, flipping, and zooming to increase the diversity of training samples.

**2. Ensemble Model Architecture**

The model combines ResNet101V2 and VGG19 pre-trained architectures:

Base Models: ResNet101V2 and VGG19 are loaded with ImageNet weights and fine-tuned for the classification task.

Custom Layers: Global Average Pooling 2D layers, Dense layers with ReLU activation, and Dropout layers are added to reduce overfitting.

Ensemble Strategy: The outputs of both models are merged, and a final Dense layer with softmax activation is used for classification.

**3. Training Process**

Loss Function: Categorical Crossentropy is used as the loss function for multi-class classification.

Optimizer: The Adam optimizer is employed for efficient weight updates.

Callbacks:

EarlyStopping: Monitors validation loss and stops training if it does not improve for a specified number of epochs.

ReduceLROnPlateau: Reduces the learning rate when validation loss plateaus.

**4. Model Evaluation**

The trained model is evaluated using:

Accuracy: Overall classification accuracy.

Confusion Matrix: Provides insights into model predictions.

Precision, Recall, and F1-Score: Measure the model's performance across individual classes.

**Results**

The ensemble model achieves high accuracy and demonstrates superior performance in classifying diseases compared to individual architectures. The detailed metrics and confusion matrix are provided in the notebook output.
