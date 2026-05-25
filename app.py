import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile
from PIL import Image
import os

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "audio_classifier_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Make sure 'audio_classifier_model.h5' is in your project folder.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# Class names (MUST match training order)
class_names = ['dog_bark', 'drilling', 'engine_idling', 'siren']

# ----------------------------
# UI
# ----------------------------
st.title("🎧 Audio Classification App")
st.write("Upload a .wav file and the model will predict the sound.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

# ----------------------------
# Process File
# ----------------------------
if uploaded_file is not None:

    st.info("Processing audio...")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    try:
        # Load audio
        y, sr = librosa.load(temp_audio_path)

        # Create spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot spectrogram
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')

        st.subheader("📊 Spectrogram")
        st.pyplot(fig)

        # Save spectrogram image temporarily
        temp_img_path = "temp_spectrogram.png"
        fig.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load image for model
        img = Image.open(temp_img_path).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

        # Show result
        st.subheader("🎯 Prediction")
        st.success(f"Predicted Sound: {predicted_class}")

        st.write(f"Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
