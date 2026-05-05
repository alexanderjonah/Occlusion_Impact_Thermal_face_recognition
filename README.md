# Occlusion-Resilient Thermal Face Recognition
This project explores thermal face recognition under occlusion, focusing on identifying individuals even when facial regions such as glasses or masks are obstructed.
# Overview
Thermal imaging is robust to lighting variations but struggles with occlusions. This project evaluates three deep learning architectures from different generations to determine which performs best and which is most resilient under occluded conditions.
# Models Used

ResNet50 (classic CNN baseline)
EfficientNet-B0 (optimized CNN)
ConvNeXt-Tiny (modern transformer-inspired CNN)

# Dataset

Organized into Train / Validate / Test
8 subject classes (thermal images only)
Test set split into Non-Occluded and Occluded conditions for separate evaluation

# Features

Multi-model training with transfer learning and selective layer freezing
Early stopping and weight decay to prevent overfitting
Accuracy and performance drop comparison across architectures
Precision, Recall and F1-Score evaluation
Grad-CAM visualization for explainability

# Results
Models are compared based on clean accuracy, occluded accuracy, and performance drop. EfficientNet-B0 achieves the best overall accuracy while ConvNeXt-Tiny shows the strongest robustness under occlusion with the smallest performance drop.
# Explainability
Grad-CAM is used to visualize which facial regions each model focuses on, and how attention shifts when the face is partially occluded.
# Tech Stack

Python
PyTorch / TorchVision
OpenCV
scikit-learn
Matplotlib

# Goal
To identify the most effective and occlusion-resilient deep learning architecture for real-world thermal face recognition.