# Facial_recognition
# Facial Expression Recognition using CNN (FER2013)
A deep learning project to detect human emotions from facial images using a CNN trained on the FER2013 dataset.

# 📌 Features
- Real-time webcam emotion detection using OpenCV
- 7 emotion classes:
  Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Built using TensorFlow & Keras
- Preprocessing with ImageDataGenerator

## 📂 Dataset Structure
train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/
test/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/

    ## ⚙️ Installation
pip install tensorflow opencv-python numpy pandas scikit-learn


## ▶️ How to Run
**1. Train the Model:**
python train_model.py

**2. Run Real-Time Emotion Detection:**
python detect_emotion_webcam.py


## 📊 Model Architecture
- Conv2D → MaxPooling2D
- Conv2D → MaxPooling2D
- Flatten
- Dense (256 neurons) → Dropout (0.5)
- Dense (7 neurons, Softmax)

## 📈 Training Details
- Optimizer: Adam
- Loss: categorical_crossentropy
- Epochs: 10
- Batch Size: 64


## 📦 Future Enhancements
- Add data augmentation for better generalization
- Deploy as Flask/Streamlit web app
- Use Mediapipe for faster face detection


