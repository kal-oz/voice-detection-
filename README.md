# 🎧 Audio Classification باستخدام CNN

##  Project Overview

This project is an Audio Classification System that can recognize different environmental sounds such as:

*  Dog Bark
*  Drilling
*  Engine Idling
*  Siren

The system converts audio into Mel Spectrograms and uses a Convolutional Neural Network (CNN) to classify the sounds.

---

##  How It Works

The pipeline of the project:

Audio (.wav)
   ↓
Convert to Spectrogram (Image)
   ↓
CNN Model (TensorFlow/Keras)
   ↓
Prediction (Sound Class)

---

##  Project Structure

audio-classification-project/
│
├── data/
│   └── UrbanSound8K/
│
├── spectrograms/
│   ├── dog_bark/
│   ├── drilling/
│   ├── engine_idling/
│   └── siren/
│
├── audio_classifier_model.h5
├── convert_to_spectrogram.py
├── load_data.py
├── model.py
├── train.py
├── app.py
└── README.md

---

##  Technologies Used

* Python 
* TensorFlow / Keras 
* Librosa 🎧 (audio processing)
* Matplotlib 
* Streamlit  (UI)

---

## 🚀 How to Run the Project

### 1. Install dependencies

pip install numpy matplotlib librosa scikit-learn tensorflow streamlit

---

### 2. Train the model

python train.py

---

### 3. Run the app

streamlit run app.py

---

### 4. Use the App

 Upload a .wav file
 View the spectrogram
 Get prediction + confidence

---

## Dataset

 UrbanSound8K Dataset
 Used selected classes:

  dog_bark
  drilling
  engine_idling
  siren

---

## Results

The model is capable of:

Converting audio into spectrograms
Learning sound patterns
Predicting sound classes with reasonable accuracy

---

## Acknowledgment

This project demonstrates a complete end-to-end machine learning pipeline, from data processing to deployment.

---

## 🎉 Status

Project Completed Successfully
