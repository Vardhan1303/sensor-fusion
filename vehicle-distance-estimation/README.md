# 🚘 Task 1: Vehicle Distance Estimation 🛣️

This project focuses on estimating the distance of vehicles using sensor fusion techniques. It integrates data from LiDAR, Radar, and Camera sensors to enhance distance estimation accuracy.

---

## 🎯 Overview

- **Objective**: Improve distance estimation accuracy for vehicles using sensor fusion.
- **Sensors Used**: LiDAR, Radar, and Camera.
- **Key Techniques**:
  - 🖼️ Vehicle detection using YOLOv5.
  - 📊 Integration of LiDAR and Radar data for distance estimation.
  - 🔄 Application of a Kalman filter for sensor fusion.

---

## 📂 Dataset

The project uses the `dataset_astyx_hires2019` dataset, with a focus on image `000131` for analysis and evaluation.

---

## 🛠️ Methodology

1. **Vehicle Detection**: YOLOv5 is used for real-time vehicle detection in images.
2. **Data Integration**: LiDAR and Radar data are projected onto camera images for visualization and analysis.
3. **Distance Estimation**: LiDAR and Radar points are filtered and processed to estimate distances.
4. **Sensor Fusion**: A Kalman filter is applied to fuse LiDAR and Radar measurements for improved accuracy.

---

## 🗂️ Code Structure

```
vehicle-distance-estimation/
├── fusion.py                 # YOLOv5 vehicle detection
└── README.md                 # Task documentation
└── requirements.txt          # Dependencies
```

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Vardhan1303/sensor-fusion.git
    ```
2. Navigate to the task folder:

   ```bash
   cd sensor-fusion/vehicle-distance-estimation
    ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
    ```

4. Run the code:

   ``` bash
   python fusion.py
    ```
---

## 📊 Results

- The fusion algorithm achieved improved distance estimation accuracy compared to individual sensor measurements.
- Detailed performance analysis is provided in the jupyternotebook. 

---

## 📜 References

1. **Dataset**:

- `dataset_astyx_hires2019` – Astyx HiRes2019 Dataset: https://www.astyx.com/fileadmin/redakteur/dokumente/Astyx_Dataset_HiRes2019_specification.pdf

2. **Object Detection Model**:

- `YOLOv5` – https://github.com/ultralytics/yolov5

3. **Sensor Fusion Techniques**:

- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," Transactions of the ASME–Journal of Basic Engineering, 1960.

4. **Python Libraries**:

- **OpenCV**: https://opencv.org
- **NumPy**: https://numpy.org
- **SciPy**: https://scipy.org