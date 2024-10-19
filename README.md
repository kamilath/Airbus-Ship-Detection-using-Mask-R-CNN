# Airbus-Ship-Detection-using-Mask-R-CNN
This project is part of the Airbus Ship Detection Challenge, where we detect and segment ships in satellite images using Mask R-CNN. The Mask R-CNN architecture is trained and fine-tuned to identify ships in aerial images, with a focus on high accuracy for detection and segmentation.
# Project Overview
This repository provides a solution for detecting ships in satellite images using the Mask R-CNN model. The dataset used in this project is provided by Airbus, and it contains thousands of images, each annotated with ship masks. We implemented Mask R-CNN to perform both object detection and segmentation.

# Key features of the project:
Ship segmentation from aerial images using Mask R-CNN.
Implementation of RLE encoding/decoding for the dataset.
Training and evaluation on high-resolution images (768x768).
Visualization of ship masks, and encoded/decoded results.

# Dataset
The dataset used for this project is from the Airbus Ship Detection Challenge. It contains images along with the corresponding ship masks in Run-Length Encoded (RLE) format.
Training Data: 192,556 satellite images.
Test Data: 156,060 satellite images.
Each image is 768x768 pixels.
You can download the dataset from [Kaggle's Airbus Ship Detection Challenge.]
