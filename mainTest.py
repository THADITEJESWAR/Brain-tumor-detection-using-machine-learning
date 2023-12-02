import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')
file_path = r'D:\brainTumor\pred\pred5.jpg'

# Try to read the image
image = cv2.imread(file_path)
img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)
input_img = np.expand_dims(img, axis=0)

# Assuming the model has a sigmoid activation in the output layer for binary classification
raw_predictions = model.predict(input_img)
predicted_class = int(raw_predictions[0][0] > 0.5)  # Reverse the condition

print("Predicted class:", predicted_class)
