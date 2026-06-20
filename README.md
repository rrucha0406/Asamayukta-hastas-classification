# Asamayukta-hastas-classification
my first mediapipe project
dataset :[https://www.kaggle.com/datasets/sauravsm/kathakali-mudra-dataset]

# Bharatanatyam Mudra Recognition using MediaPipe & Machine Learning

#Project Overview

This project detects and classifies Bharatanatyam hand mudras (Asamyuta Hastas) using:

* Computer Vision (OpenCV)
* MediaPipe Hand Tracking
* Machine Learning (Random Forest)

It works in real-time using webcam input.

---

## Features

* Detects 21 hand landmarks using MediaPipe
* Converts landmarks into 63 numerical features
* Trains a gesture classification model
* Performs real-time mudra recognition via webcam

---

## Pipeline

```
Images
   ↓
MediaPipe (21 landmarks)
   ↓
Feature Extraction (63 features)
   ↓
Machine Learning Model
   ↓
Real-Time Prediction
```

---

## Project Structure

```
mudra_recognition/
│
├── dataset/
│     └── raw/
│         ├── train/
│         ├── test/
│         └── valid/
│
├── notebooks/
│     ├── 01_data_cleaning.ipynb
│     ├── 02_landmark_extraction.ipynb
│     └── 03_model_training.ipynb
│
├── models/
│     ├── mudra_model.pkl
│     └── label_encoder.pkl
│
├── app/
│     └── realtime_detection.py
│
└── README.md
```

---

## Installation

### 1. Create Environment

```
conda create -n opencv_env python=3.10 -y
conda activate opencv_env
```

### 2. Install Dependencies

```
pip install opencv-python mediapipe scikit-learn pandas matplotlib joblib
```

---

## Dataset

Dataset sourced from Kaggle:

Bharatanatyam Asamyuta Hasta Mudras Dataset

Download manually and place inside:

```
dataset/raw/
```

---

## Model Training

Run notebooks in order:

```
01_data_cleaning.ipynb
02_landmark_extraction.ipynb
03_model_training.ipynb
```

---

## Real-Time Detection

Run the application:

```
python app/realtime_detection.py
```

Press **Q** to exit.

---

## Results

Achieved strong performance using:

* Landmark normalization
* Random Forest classifier

---

## Future Improvements

* Add angle-based features
* Explore deep learning models (CNN / LSTM)
* Improve user interface
* Deploy as a web application

---

## Key Learnings

* MediaPipe landmark extraction
* Feature engineering for gesture recognition
* Real-time machine learning systems
* End-to-end ML pipeline development

---

## Author

Developed as part of an AIML portfolio project.

