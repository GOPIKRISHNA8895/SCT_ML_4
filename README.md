# ✋ Hand Gesture Recognition System

A **Machine Learning Web Application** built using **Support Vector Machine (SVM)** and **HOG (Histogram of Oriented Gradients)** feature extraction to accurately recognize and classify different hand gestures from images.

This project was developed as part of the **SkillCraft Technology Machine Learning Internship (Task-04).**

---

# 📌 Project Overview

The **Hand Gesture Recognition System** is a Computer Vision-based application that automatically identifies hand gestures from uploaded images using traditional Machine Learning techniques.

The project is implemented in two phases:

## 📓 Phase 1: Jupyter Notebook

The complete Machine Learning pipeline was developed and trained in Jupyter Notebook.

### 🔹 Steps Performed

* Dataset loading and preprocessing
* Image resizing and grayscale conversion
* Feature extraction using HOG descriptors
* Feature scaling using StandardScaler
* SVM model training
* Model evaluation and testing
* Classification report generation
* Confusion matrix visualization
* Prediction on sample gesture images

---

## 🌐 Phase 2: Streamlit Web Application

The trained SVM model was deployed into an interactive Streamlit web application.

### 🔹 Application Features

* Hand gesture image upload
* Real-time gesture prediction
* Confidence score visualization
* Gesture probability distribution chart
* Interactive dashboard UI
* Responsive design

---

# 🚀 Features

## 📸 Image Upload

* Upload JPG / JPEG / PNG images
* Instant image preview
* Easy-to-use interface

## 🤖 Gesture Recognition

* Detects hand gestures automatically
* Uses trained SVM model
* Fast prediction results

## 📊 Confidence Score

* Displays prediction confidence percentage
* Progress bar visualization
* Model certainty indication

## 📈 Probability Analysis

* Bar chart showing probabilities for all gesture classes
* Visual comparison of prediction confidence

## 🌐 Interactive Dashboard

* Streamlit-based web application
* Sidebar navigation
* Professional UI design
* Responsive layout

---

# 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Scikit-Image (HOG Features)
* Streamlit
* Joblib
* Pillow

---

# 🤖 Machine Learning Model

## 🔹 Algorithm Used

**Support Vector Machine (SVM)**

## 🔹 Feature Extraction

**HOG (Histogram of Oriented Gradients)**

## 🔹 Workflow

1. Image Input
2. Image Preprocessing
3. Grayscale Conversion
4. HOG Feature Extraction
5. Feature Scaling
6. SVM Classification
7. Gesture Prediction
8. Result Visualization

---

# 📂 Dataset

## Dataset Used

**LeapGestRecog Dataset**

The dataset contains thousands of hand gesture images collected from multiple users performing various gestures.

### Gesture Classes

* Palm
* L
* Fist
* Fist Moved
* Thumb
* Index
* OK
* Palm Moved
* C
* Down

---

# 📊 Model Evaluation

The model was evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Prediction Testing

### Visualizations Included

* Class Distribution Graph
* Sample Gesture Images
* Confusion Matrix Heatmap
* Prediction Probability Charts

---

# 📸 Application Pages

## ✋ Dashboard

* Project Overview
* Technology Stack
* Model Information
* Performance Metrics
* Gesture Recognition Banner

## 🎯 Prediction Page

* Upload Gesture Image
* View Predicted Gesture
* Confidence Score Display
* Progress Bar
* Probability Distribution Graph

## 📚 About Project

* Dataset Information
* Machine Learning Workflow
* Technology Stack
* Future Improvements

---

# ▶️ How to Run

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SCT_ML_04.git
```

## 2️⃣ Move to Project Folder

```bash
cd SCT_ML_04
```

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Streamlit App

```bash
python -m streamlit run app.py
```

---

# 📊 Learning Outcomes

This project helped in understanding:

* Image Processing Techniques
* Feature Extraction using HOG
* Support Vector Machine (SVM)
* Computer Vision Fundamentals
* Multi-Class Classification
* Model Training & Evaluation
* Streamlit Web Application Development
* End-to-End ML Deployment

---

# 🔮 Future Enhancements

## 🤖 Advanced Machine Learning

* Random Forest Comparison
* XGBoost Classification
* Ensemble Learning

## 📷 Computer Vision Improvements

* Real-Time Webcam Gesture Detection
* MediaPipe Hand Tracking Integration
* Background Removal
* Gesture Sequence Recognition

## 🧠 Deep Learning Upgrade

* CNN-Based Gesture Recognition
* Transfer Learning
* MobileNet Integration
* Real-Time Deep Learning Inference

## ☁️ Deployment

* Streamlit Cloud Deployment
* AWS Deployment
* Docker Containerization
* REST API Development

## 📊 UI Enhancements

* Plotly Interactive Charts
* Dark/Light Theme Toggle
* Mobile-Friendly Design
* Gesture History Tracking

---

# 🎯 Internship Task

**SkillCraft Technology – Machine Learning Internship**

### Task-04: Hand Gesture Recognition using Machine Learning

Develop a Hand Gesture Recognition model that can accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

---

# 👨‍💻 Developer

**Gopi Krishna**
Machine Learning Intern – SkillCraft Technology

🔗 **GitHub:[https://github.com/GOPIKRISHNA8895/SCT_ML_4.git]

🌐 **Live Demo: [https://mlhandgesturerecognition04.streamlit.app/]

---

