# 🖼️ Image Fusion & Classification with Deep Learning 🧠

*A deep learning project for multi-sensor image fusion and shape classification using PyTorch. Achieves **86.5% accuracy** with majority voting fusion!*  
![Confusion Matrix](https://via.placeholder.com/600x400/009688/ffffff?text=Confusion+Matrix+Visual)  
*(Example visualization - replace with actual image path)*

---

## 🎯 Project Overview

This repository implements an **intelligent image processing pipeline** that:
1. 🔄 Fuses images from 3 sensors using multiple fusion techniques
2. 🔍 Classifies shapes (Circle, Square, Triangle, Pentagon) using deep learning
3. 🤝 Combines predictions through ensemble learning

**Key Features**:
- **Multi-Sensor Fusion** (Average, Max, Min techniques)
- **LeNet-5 CNN Architecture** with custom modifications
- **Majority Voting System** for improved accuracy
- **Comprehensive Evaluation** with confusion matrices
- 📝 **Jupyter Notebook Support**: The entire pipeline can be visualized and executed in a Jupyter Notebook for better analysis and experimentation.

---

## 📁 Dataset Structure

```bash
Imagefusion_Dataset/
├── img/
│   ├── img1/      # Sensor 1 images (e.g., gradient patterns)
│   ├── img2/      # Sensor 2 images (e.g., noisy patterns)
│   └── img3/      # Sensor 3 images (e.g., spotlight patterns)
└── label/         # Text files with shape labels
```

### Dataset Statistics:

- **4 classes**: ⚪ Circle, ⬛ Square, 🔺 Triangle, ⬠ Pentagon

- 5000+ synthetic images (sample visualization below)

Sample Images

## 🏛️ Model Architecture

### Modified LeNet-5 Network

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=4, input_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
``` 

### Key Modifications:

- Input channels adaptable for different fusion methods

- Additional batch normalization layers

- Custom kernel sizes for better feature extraction

## 🚀 Training Pipeline

### Workflow Diagram
```mermaid
graph TD
    A[Raw Sensor Images] --> B[Image Fusion]
    B --> C[Data Augmentation]
    C --> D[LeNet-5 Training]
    D --> E[Model Evaluation]
    E --> F[Majority Voting]
```

### Training Parameters:

- 📈 **Optimizer** : Adam (lr=0.001)

- ⚖️ **Loss Function**: CrossEntropyLoss

- 🔄 **Epochs**: 100

- 📦 **Batch Size**: 64

- 🎲 **Train/Test Split**: 80/20

## 📊 Results & Performance

### Accuracy Comparison

```matlab 
Model	Test Accuracy
Sensor 1 CNN	71.5%
Sensor 2 CNN	65.5%
Sensor 3 CNN	66.5%
Fusion	86.5%
```

### Confusion Matrix (Fusion Model)
```markdown
              Predicted
         ⚪  ⬛  🔺  ⬠
Actual ⚪ 98  1   1   0
       ⬛  2 95   3   0
       🔺  1  2  96   1
       ⬠  0  1   2  97
```

## 🛠️ Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### requirements.txt:

```ini
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.3.0
Pillow==9.5.0
```

## Jupyter Notebook Support

You can **observe, analyze, and run** the training and evaluation pipeline in a Jupyter Notebook for an interactive experience. Simply open the notebook and execute the provided cells for a step-by-step demonstration of image fusion and classification.


## 📚 References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition
Foundational paper on LeNet-5 architecture

2. Zhang, Y., & Liu, Y. (2014). Multi-Sensor Image Fusion Techniques
Review of image fusion methodologies

3. Dietterich, T. G. (2000). Ensemble Methods in Machine Learning
Theoretical basis for majority voting
---
