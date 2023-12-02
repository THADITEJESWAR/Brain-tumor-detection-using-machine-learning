from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and resize images from a folder
def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_resized = img.resize(target_size)  # Resize to a common size
        img_array = np.array(img_resized).flatten()  # Flatten the image
        images.append(img_array)
        labels.append(1 if "yes" in folder else 0)
    return images, labels

# Use raw string literals for paths
yes_folder = r'D:\brainTumor\datasets\yes'
no_folder = r'D:\brainTumor\datasets\no'
pred_folder = r'D:\brainTumor\pred'

# Load, resize, and flatten images for training
images_yes, labels_yes = load_and_resize_images(yes_folder)
images_no, labels_no = load_and_resize_images(no_folder)

# Combine data for training
X_train = np.concatenate((images_yes, images_no), axis=0)
y_train = np.concatenate((labels_yes, labels_no))

# Train Lasso regression model with scaled training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Lasso regression model
lasso_reg = Lasso(alpha=0.1)  # Adjust the alpha parameter as needed
lasso_reg.fit(X_train_scaled, y_train)

# Load, resize, and flatten images for testing
X_test, y_test = load_and_resize_images(pred_folder)

# Scale the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions on test set
y_pred = lasso_reg.predict(X_test_scaled)

# Convert y_test to a NumPy array
y_test = np.array(y_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print Performance Metrics
print(f"Mean Squared Error (MSE): {mse}")

# Convert predicted values to binary (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
y_pred_binary = np.where(y_pred >= threshold, 1, 0)

# Print Classification Metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, zero_division=1)
recall = recall_score(y_test, y_pred_binary, zero_division=1)
f1 = f1_score(y_test, y_pred_binary)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print(f"Mean Squared Error (MSE): {mse}")

# Print Predicted Values
print("\nPredicted Values:")
print(y_pred_binary)

# Visualization of Predictions
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual Labels')
ax.set_ylabel('Predicted Labels')
ax.set_title('Lasso Regression Predictions vs Actual Labels')
plt.show()
