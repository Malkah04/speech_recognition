# Speech Emotion Recognition (SER) 🎙️

## Project Description

This project aims to build a Speech Emotion Recognition (SER) system that classifies human emotions from speech audio recordings using machine learning techniques. Audio features are extracted using Librosa, and an MLPClassifier from Scikit-learn is used for emotion classification.

---

## Objective

* Extract audio features from speech signals
* Train a machine learning model to recognize emotions
* Evaluate model performance

---

## Dataset

* CREMA-D
The dataset will include at least five emotion classes.

---

## Feature Extraction

The following features will be extracted using Librosa:

* MFCCs (20 coefficients)
* Chroma features
* Mel Spectrogram
* Zero-Crossing Rate
* Spectral Centroid

All extracted features will be combined into a single feature vector.

---

## Model

* MLPClassifier
* StandardScaler for feature scaling
* Train/Test split

---

## Evaluation

The model will be evaluated using:

* Accuracy
* Classification report
* Confusion matrix

---
