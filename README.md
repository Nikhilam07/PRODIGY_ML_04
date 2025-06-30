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

----------------------------------------------
🧠 Model Training (Run in Google Colab)
----------------------------------------------
1. Download the LeapGestRecog dataset from Kaggle
2. Upload the zip file to Google Colab and extract it
3. Train a CNN on grayscale images (100x100)
4. Save the trained model using:

   model.save("gesture_cnn_model.h5")

5. Download the model file to your computer:

   from google.colab import files  
   files.download("gesture_cnn_model.h5")

----------------------------------------------
💻 Run the Model Locally with Webcam (PC)
----------------------------------------------
1. Create a folder on your PC and place these files inside:

   - gesture_cnn_model.h5         (your trained model)
   - gesture_webcam_demo.py       (script for real-time webcam prediction)
   - requirements.txt             (dependencies list)
   - README.md                    (project documentation)
   - instructions.txt             (this file)

2. Open Terminal or Command Prompt in that folder

3. Install required libraries (run only once):

   pip install -r requirements.txt

4. Run the real-time gesture prediction:

   python gesture_webcam_demo.py

5. A webcam window will open with a green rectangle.
   Place your hand gesture inside the box and view live predictions.
   Press 'q' to exit.

----------------------------------------------
🧾 File Overview
----------------------------------------------
gesture_cnn_model.h5       - Trained model file (Keras/TensorFlow)  
gesture_webcam_demo.py     - Python script to run webcam inference  
README.md                  - Full project overview and GitHub documentation  
requirements.txt           - Python libraries required for this project  
instructions.txt           - Step-by-step usage guide  

----------------------------------------------
⚠️ Notes
----------------------------------------------
- Make sure your webcam is connected and working.
- The model expects grayscale 100x100 images as input.
- Lighting and background can affect accuracy.
- If the model always predicts the same class, double-check the ROI and preprocessing.

----------------------------------------------
✅ Optional Enhancements
----------------------------------------------
- Use MediaPipe for hand detection instead of static ROI
- Add GUI using Tkinter or Streamlit
- Control mouse, volume, or applications using gestures

----------------------------------------------
📜 License & Credits
----------------------------------------------
Dataset: LeapGestRecog by GTI-UPM  
Link: https://www.kaggle.com/datasets/gti-upm/leapgestrecog  
This code is for educational and research purposes only.
