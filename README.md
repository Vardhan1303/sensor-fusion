# 🚗 Sensor Fusion Projects 🚀

Welcome to the **Sensor Fusion Projects** repository! This repository contains two exciting projects focused on **Vehicle Distance Estimation** (Task 1) and **Fusion for Classification** (Task 2). These projects explore the integration of data from multiple sensors (e.g., LiDAR, Radar, and Cameras) to improve object detection, distance estimation, and classification accuracy.

---

## 📂 Projects

### 1. 🚘 **Task 1: Vehicle Distance Estimation**
   - **Description**: This project focuses on estimating the distance of vehicles using sensor fusion techniques. It integrates data from LiDAR, Radar, and Camera sensors to enhance distance estimation accuracy.
   - **Key Features**:
     - 🖼️ Vehicle detection using YOLOv5.
     - 📊 Integration of LiDAR and Radar data for distance estimation.
     - 🔄 Application of a Kalman filter for sensor fusion.
   - **Folder**: [vehicle-distance-estimation](vehicle-distance-estimation)

### 2. 🎯 **Task 2: Fusion for Classification**
   - **Description**: This project explores fusion strategies (Low-Level and High-Level Fusion) to improve the classification accuracy of geometric shapes using a synthetic dataset.
   - **Key Features**:
     - 🧠 Adaptation of the LeNet-5 model for shape classification.
     - 🔧 Implementation of Low-Level and High-Level Fusion techniques.
     - 📈 Comparative analysis of fusion strategies.
   - **Folder**: [fusion-for-classification](fusion-for-classification)

---

## 🗂️ Repository Structure

```
sensor-fusion/
├── vehicle-distance-estimation/             # Task 1: Vehicle Distance Estimation
│ ├── README.md                              # Task 1 documentation
│ ├── fusion.ipynb                           # Code files for Task 1
│ └── requirements.txt                       # Dependencies 
├── fusion-for-classification/               # Task 2: Fusion for Classification
│ ├── README.md                              # Task 2 documentation
│ ├── classification.ipynb                   # Code files for Task 2
│ ├── Imagefusion_dataset/                   # Dataset 
│ └── requirements.txt                       # Dependencies
└── README.md                                # Main repository documentation
```
---

## 🚀 Getting Started

1. Clone the repository:
``` bash
   git clone https://github.com/Vardhan1303/sensor-fusion.git
```

2. Navigate to the respective task folder for detailed instructions and code.

---

## 📦 Dependencies

- Python 3.x
- Libraries: PyTorch, OpenCV, NumPy, Pandas, Matplotlib, etc. (See individual task READMEs for specific requirements.)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🎓 Academic Context

These projects were developed as part of the **Sensor Fusion** course during the **Master of Sciences in Mechatronics** program at the **University of Applied Sciences Ravensburg-Weingarten**. The work was conducted under the guidance of **Mr. Felix Berens**.

---

⭐ If you like this project, give it a star! 🌟