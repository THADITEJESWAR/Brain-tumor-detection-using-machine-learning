from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to load and resize images from a folder
def load_and_resize_images(folder, target_size=(100, 100), label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_resized = img.resize(target_size)  # Resize to a common size
        img_array = np.array(img_resized).flatten()  # Flatten the image
        images.append(img_array)
        labels.append(label)  # Use the provided label for the test set
    return images, labels

# Use raw string literals for paths
yes_folder = r'D:\brainTumor\datasets\yes'
no_folder = r'D:\brainTumor\datasets\no'
pred_folder = r'D:\brainTumor\pred'  # Adjust this path accordingly

# Load, resize, and flatten images for training
images_yes, labels_yes = load_and_resize_images(yes_folder, label=1)
images_no, labels_no = load_and_resize_images(no_folder, label=0)

# Combine data for training
X_train = np.concatenate((images_yes, images_no), axis=0)
y_train = np.concatenate((labels_yes, labels_no))

# Train linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on test set
X_test, y_test = load_and_resize_images(pred_folder, label=1)  # Assuming label 1 for "yes" predictions
y_pred = linear_reg.predict(X_test)

# Convert predictions to binary values based on a threshold (e.g., 0.5)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate the performance on the test set
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Print the performance metrics
print("Performance on Test Set:")
print(f"Mean Squared Error: {mse}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Flatten X_test if it is 2D
X_test_flat = np.array(X_test).flatten()

# Ensure X_test_flat and y_test have the same length
min_length = min(len(X_test_flat), len(y_test))
X_test_flat = X_test_flat[:min_length]
y_test = y_test[:min_length]


# Visualization of Performance Metrics
metrics_labels = ['Mean Squared Error', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [mse, accuracy, precision, recall, f1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(metrics_labels, metrics_values, color=['blue', 'green', 'orange', 'red', 'purple'])
ax.set_ylabel('Metric Value')
ax.set_title('Performance Metrics on Test Set')
plt.show()

# Plotting the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(X_test_flat, y_test, color='blue', label='Actual')
plt.scatter(X_test_flat, y_pred, color='red', label='Predicted')
plt.plot(X_test_flat, y_pred, color='green', linewidth=2, label='Best Fit Line')

plt.title('Linear Regression: Best Fit Line')
plt.xlabel('Feature (Flattened Image Pixel Values)')
plt.ylabel('Target Variable (Label)')
plt.legend()
plt.show()
