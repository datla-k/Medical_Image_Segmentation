# Medical_Image_Segmentation

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org) [![](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)


## Introduction 

This project is an implementation of the U-Net++ model for medical image segmentation mainly to detect Polyp which is a projecting growth of tissue from a surface in the body, usually a mucous membrane.

<img src = "https://github.com/datla-k/Medical_Image_Segmentation/blob/main/Images/polyp.jpg"/>


## Data Overview
The CVC-Clinic database  is a collection of digital medical images extracted from colonoscopy videos that contains several examples of Polyp frames and corresponding ground truth.

## Aim of the Project
The aim of this project is to develop a deep learning model for medical image segmentation using U-Net architecture to Segement the polyps from Colonoscopy images

## Approach Towards the Implementation

1. Understading the essence of the dataset.
2. Understanding the evaluation metrics used for the prediction.
3. Unet vs Unet++
4. Data Augmentation
5. Model Building using Pytorch
6. Model Training
7. Model Prediction

## Pipeline

### 1. Dataset.py

Responsible for loading the images and their corresponding segmentation masks, applying transformations(Resizing and Normalisation), preparing the data for model training and evaluation

(torch.utils.data.Dataset)
This DataSet class is a crucial component of the ML pipeline for medical image segmentation. It abstracts away the complexities of data loading and preprocessing, ensuring that the data is in the correct format and ready for model training or evaluation. The optional transform parameter allows for flexible data augmentation and preprocessing, making the dataset adaptable to different models and training scenarios.

### 2. Network.py

The network.py file encapsulates the architectural definition of the U-Net++ model, leveraging nested convolutional blocks for enhanced feature extraction and segmentation accuracy. This architecture is particularly suited for medical image segmentation tasks, where precise delineation of anatomical structures or pathological findings is critical. The option for deep supervision enables more effective training by encouraging feature learning at multiple scales.

### 3. Predict.py

The predict.py script sets up the necessary preprocessing steps for images before they are fed into a U-Net++ model for segmentation prediction. It demonstrates the use of albumentations for image transformation, ensuring images are correctly sized and normalized according to the model's expectations.

### 4. Train.py

This script encapsulates the training logic for a U-Net++ model, focusing on the medical image segmentation task. It handles both standard and deep supervision training modes, calculates loss and IoU metrics to track model performance, and updates model weights based on the computed gradients. The use of AverageMeter for tracking metrics and a progress bar for real-time feedback makes monitoring the training process straightforward and effective.

### 5. Utils.py

Essential for monitoring model performance (AverageMeter) and evaluating segmentation accuracy (iou_score). The AverageMeter allows for easy tracking of metrics over epochs, while the iou_score function provides a direct measure of model effectiveness in terms of accurately predicting the segmentation masks compared to the ground truth. These utilities support effective model training, tuning, and evaluation processes by offering a straightforward way to measure and report performance.

### 6. Validate.py

Evaluates the model's performance on unseen data, providing insights into how well the model generalizes beyond the training data. By comparing the model's loss and IoU on the validation set to those on the training set, developers can diagnose issues such as overfitting. This function, alongside the training function, forms a critical component of the model development and evaluation pipeline, allowing for iterative improvements to model architecture, training procedures, and hyperparameters based on empirical performance metrics.