
import cv2
import numpy as np

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 48, 48, 1) / 255.0
    return img
