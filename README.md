# Semantic Segmentation using DeepLabV3

This repository contains a custom implementation of **DeepLabV3** for semantic segmentation tasks. The model uses a custom-built **Atrous Spatial Pyramid Pooling (ASPP)** module and a **ResNet101** backbone, and is trained on the **Pascal VOC 2012** dataset.

## Overview
DeepLabV3 is a state-of-the-art deep learning model designed for semantic segmentation, capable of accurately labeling each pixel in an image into predefined categories. This implementation focuses on achieving high performance in semantic segmentation tasks using dilated convolutions and multi-scale context aggregation.

## Features
- **DeepLabV3**: A robust model architecture designed for pixel-level classification.
- **Custom-built ASPP**: Captures multi-scale context using atrous (dilated) convolutions.
- **ResNet101 Backbone**: Provides powerful feature extraction capabilities with a 101-layer deep residual network.
- **Pascal VOC 2012**: Trained and evaluated on the Pascal VOC 2012 dataset, which contains 20 object categories and one background class.

## Model Architecture
- **Backbone**: ResNet101 pre-trained on ImageNet.
- **ASPP Module**: Custom-built to capture multi-scale information using different dilation rates.
- **Decoder**: Upsamples the feature maps to match the original input resolution for precise pixel-wise classification.

## Dataset
The model is trained using the **Pascal VOC 2012** dataset, which includes 1,464 images for training, 1,449 images for validation, and 1,456 images for testing, across 20 object categories.

## Getting Started

### Prerequisites
1. Install Python 3.x and the required dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/pojesh/semantic-segmentation-deeplabv3.git
    cd semantic-segmentation-deeplabv3
    ```

### Training the Model
1. Download the Pascal VOC 2012 dataset from the official site and place it in the `data/` folder.
2. Train the model:
    ```bash
    python process.ipynb --dataset ./data/VOCdevkit/VOC2012 --epochs 50 --batch-size 16
    ```
