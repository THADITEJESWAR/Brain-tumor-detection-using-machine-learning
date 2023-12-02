import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to calculate and print evaluation metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

# Load the model
model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')

# Load the test dataset and labels
image_directory = 'datasets/'
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

label_no_tumor = [0] * len(no_tumor_images)
label_yes_tumor = [1] * len(yes_tumor_images)

# Balance the test set
no_tumor_test_samples = np.random.choice(no_tumor_images, size=len(yes_tumor_images), replace=False)

# Combine both classes for the test set
test_samples = yes_tumor_images + list(no_tumor_test_samples)
test_labels = label_yes_tumor + label_no_tumor[:len(yes_tumor_images)]

# Load and preprocess the test set
dataset_test = []

for i, image_name in enumerate(test_samples):
    if image_name.split('.')[1] == 'jpg':
        if i < len(yes_tumor_images):
            image = cv2.imread(image_directory + 'yes/' + image_name)
        else:
            image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset_test.append(np.array(image))

dataset_test = np.array(dataset_test)
x_test = normalize(dataset_test, axis=1)
y_test = to_categorical(test_labels, num_classes=2)

# Make predictions and evaluate the model
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert one-hot encoded labels back to binary
y_test_binary = np.argmax(y_test, axis=1)

# Evaluate the model
evaluate_model(y_test_binary, y_pred)

# Confusion Matrix
conf_mat = confusion_matrix(y_test_binary, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
