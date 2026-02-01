import sys
print(sys.executable)

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load("mlp_cremad_model.pkl")
scaler = joblib.load("scaler.pkl")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# ===============================
# Feature Extraction (SAME AS TRAINING)
# ===============================
def extract_features(file_path, mfcc_n=20, mel_n=128):
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_n)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_n)
    mel_mean = np.mean(mel, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid, axis=1)

    f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    f0 = f0[~np.isnan(f0)]
    f0_mean, f0_std = (np.mean(f0), np.std(f0)) if len(f0) > 0 else (0, 0)

    rms = librosa.feature.rms(y=y)
    rms_mean, rms_std = np.mean(rms), np.std(rms)

    return np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean,
        mel_mean,
        zcr_mean,
        spec_centroid_mean,
        [f0_mean, f0_std, rms_mean, rms_std]
    ])

# ===============================
# UI
# ===============================
st.title("🎙️ Speech Emotion Recognition Dashboard")
st.markdown("Upload an audio file and let the AI detect the emotion.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Waveform
    y, sr = librosa.load("temp.wav", sr=None)
    fig, ax = plt.subplots(figsize=(10, 0.5))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Feature extraction
    features = extract_features("temp.wav")
    features_scaled = scaler.transform([features])

    # Prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # ===============================
    # Results
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Predicted Emotion")
        st.success(prediction.upper())

    with col2:
        st.subheader("📊 Confidence")
        confidence = np.max(probabilities) * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    # Probability chart
    st.subheader("🔍 Emotion Probabilities")
    fig2, ax2 = plt.subplots(figsize=(10, 0.5))
    ax2.bar(EMOTIONS, probabilities)
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Developed for Speech Emotion Recognition Project 🎓")