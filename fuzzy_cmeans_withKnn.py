import numpy as np
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# Load, resize, and flatten images for training
images_yes, labels_yes = load_and_resize_images(yes_folder)
images_no, labels_no = load_and_resize_images(no_folder)

# Combine data for training
X_train_yes, X_test_yes, y_train_yes, y_test_yes = train_test_split(images_yes, labels_yes, test_size=0.2, random_state=42)
X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(images_no, labels_no, test_size=0.2, random_state=42)

# Combine data for training and testing
X_train = np.concatenate((X_train_yes, X_train_no), axis=0)
X_test = np.concatenate((X_test_yes, X_test_no), axis=0)
y_train = np.concatenate((y_train_yes, y_train_no))
y_test = np.concatenate((y_test_yes, y_test_no))

# Fuzzy C-Means Clustering on the training set
cntr, u_train, _, _, _, _, _ = fuzz.cluster.cmeans(X_train.T, 2, 2, error=0.005, maxiter=1000, init=None)

# Get the cluster memberships for the training set
cluster_membership_train = u_train.argmax(axis=0)

# Use KNN for classification based on FCM cluster assignments on the training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, cluster_membership_train)

# Predict using KNN based on FCM cluster assignments on the test set
u_test, _, _, _, _, _, _ = fuzz.cluster.cmeans(X_test.T, 2, 2, error=0.005, maxiter=1000, init=None)
cluster_membership_test = u_test.argmax(axis=0)
predicted_labels = knn.predict(X_test)

# Print Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, zero_division=1)
recall = recall_score(y_test, predicted_labels, zero_division=1)
f1 = f1_score(y_test, predicted_labels)
conf_matrix = confusion_matrix(y_test, predicted_labels)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize FCM clustering
plt.scatter(X_train[:, 0], X_train[:, 1], c=cluster_membership_train, cmap='viridis', s=30, edgecolors='k')
plt.title('FCM Clustering')
plt.show()

# Visualization of Predicted Outputs
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(range(1, len(predicted_labels) + 1), predicted_labels, edgecolors=(0, 0, 0))
ax.set_xlabel('Sample Index')
ax.set_ylabel('Predicted Labels')
ax.set_title('FCM + KNN Predicted Outputs')
plt.show()
