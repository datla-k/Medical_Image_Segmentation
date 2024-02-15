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
