import tensorflow as tf
import numpy as np
import cv2
import warnings
from tensorflow.keras.applications.resnet_v2 import preprocess_input

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------
# Load model
# --------------------------
model = tf.keras.models.load_model("emotion_model.keras")
print("✅ Model loaded successfully!")

# --------------------------
# Define classes
# --------------------------
classes = ['angry', 'happy', 'neutral', 'sad', 'surprised', 'fearful', 'disgusted']

# --------------------------
# Load image
# --------------------------
img_path = "test.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f" Could not read image from {img_path}")

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --------------------------
# Detect face using Haar Cascade
# --------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

if len(faces) > 0:
    # Use the first detected face
    x, y, w, h = faces[0]
    face_img = img_rgb[y:y+h, x:x+w]
else:
    # If no face detected, use the whole image
    face_img = img_rgb

# --------------------------
# Resize & preprocess face
# --------------------------
face_img = cv2.resize(face_img, (224, 224))
face_img = preprocess_input(face_img)
face_img = np.expand_dims(face_img, axis=0)

# --------------------------
# Predict emotion
# --------------------------
pred = model.predict(face_img, verbose=0)
emotion = classes[np.argmax(pred)]
confidence = tf.nn.softmax(pred).numpy()[0][np.argmax(pred)] * 100

# --------------------------
# Show result
# --------------------------
print(f" Predicted Emotion: {emotion} ({confidence:.2f}% confidence)")
