from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and resize images from a folder
def load_and_resize_images(folder, label, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_resized = img.resize(target_size)  # Resize to a common size
        img_array = np.array(img_resized).flatten()  # Flatten the image
        images.append(img_array)
        labels.append(label)
    return images, labels

# Use raw string literals for paths
yes_folder = r'D:\brainTumor\datasets\yes'
no_folder = r'D:\brainTumor\datasets\no'
pred_folder = r'D:\brainTumor\pred'

# Load, resize, and flatten images for training
images_yes, labels_yes = load_and_resize_images(yes_folder, 1)  # Label 1 for tumor
images_no, labels_no = load_and_resize_images(no_folder, 0)  # Label 0 for no tumor

# Combine data for training
X_train = np.concatenate((images_yes, images_no), axis=0)
y_train = np.concatenate((labels_yes, labels_no))

# Train logistic regression model with scaled training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression model
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train_scaled, y_train)

# Load, resize, and flatten images for testing
X_test, y_test = load_and_resize_images(pred_folder, 1)  # Label 1 for tumor

# Scale the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions on test set
y_pred = logistic_reg.predict(X_test_scaled)

# Visualization of Predictions with Scatter Plot
plt.figure(figsize=(10, 6))

# Scatter plot for actual data
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', marker='o')

# Scatter plot for predicted data
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted', marker='x')

plt.title('Scatter Plot of Actual vs Predicted Labels')
plt.xlabel('Image Index')
plt.ylabel('Label')
plt.legend()
plt.show()

# Evaluate the performance on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Performance Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Prediction on new images from "pred" folder
X_pred, _ = load_and_resize_images(pred_folder, 1)  # Label 1 for tumor
X_pred_scaled = scaler.transform(X_pred)  # Scale the new images

# Make predictions on new images
y_pred_new = logistic_reg.predict(X_pred_scaled)

# Display predictions for new images
print("\nPredictions on New Images:")
for i, prediction in enumerate(y_pred_new):
    print(f"Image {i+1}: {'Tumor' if prediction == 1 else 'No Tumor'}")
