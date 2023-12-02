import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.alexnet import preprocess_input
import os
import matplotlib.pyplot as plt

# Load and preprocess images for training
def load_and_preprocess_images(folder, target_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
        labels.append(1 if "yes" in folder else 0)
    return images, labels

# Use raw string literals for paths
yes_folder = r'D:\brainTumor\datasets\yes'
no_folder = r'D:\brainTumor\datasets\no'
pred_folder = r'D:\brainTumor\pred'

# Load, preprocess, and flatten images for training
images_yes, labels_yes = load_and_preprocess_images(yes_folder)
images_no, labels_no = load_and_preprocess_images(no_folder)

# Combine data for training
X_train = np.concatenate((images_yes, images_no), axis=0)
y_train = np.concatenate((labels_yes, labels_no))

# Flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)

# Fuzzy C-Means Clustering
# Note: Adjust parameters based on your specific requirements
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X_train.T, 2, 2, error=0.005, maxiter=1000, init=None)

# Get the cluster memberships
cluster_membership = u.argmax(axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Use KNN for classification based on FCM cluster assignments
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, cluster_membership)

# Predict using KNN based on FCM cluster assignments
predicted_labels = knn.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualization of Predicted Outputs
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(range(1, len(y_test) + 1), y_test, label='Actual Labels', edgecolors=(0, 0, 0))
ax.scatter(range(1, len(predicted_labels) + 1), predicted_labels, label='Predicted Labels', edgecolors=(0, 0, 0))
ax.legend()
ax.set_xlabel('Sample Index')
ax.set_ylabel('Labels')
ax.set_title('Actual vs Predicted Outputs')
plt.show()
