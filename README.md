# PRODIGY_ML_04
# 🤚 Hand Gesture Recognition using CNN

This project implements a **real-time hand gesture recognition system** using a Convolutional Neural Network (CNN) trained on the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog). It uses OpenCV to capture webcam input and TensorFlow to perform gesture classification live.

---

## 📸 Demo
<img src="demo.gif" width="600">

---

## 📂 Dataset
**LeapGestRecog** is a publicly available dataset from the **Gestures and Recognition group at the Technical University of Madrid (GTI-UPM)**.

- 📎 **Source**: [https://www.kaggle.com/datasets/gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- 🧑‍🤝‍🧑 10 gesture classes
- 📷 Over 20,000 grayscale images
- ✋ Captured using the Leap Motion Controller

> ✅ All credit for the dataset goes to the original creators at GTI-UPM.

---

## 🧠 Model Overview

- CNN with 3 convolutional layers + MaxPooling
- Trained on grayscale 100×100 images
- Achieved over **99% test accuracy**
- Deployed using OpenCV for real-time prediction

---

## 🧪 Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 🚀 How to Use

### 🧠 Step 1: Train the Model in Colab

1. Open the training notebook in **Google Colab**.
2. Train the model using the LeapGestRecog dataset.
3. Save the model after training:
   ```python
   model.save("gesture_cnn_model.h5")
