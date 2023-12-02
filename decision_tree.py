from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# Train Decision Tree model with hyperparameter tuning
param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
dt_classifier = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train Decision Tree model with the best hyperparameters
dt_classifier_best = DecisionTreeClassifier(**best_params, random_state=42)
dt_classifier_best.fit(X_train, y_train)

# Load, resize, and flatten images for testing
X_test, y_test = load_and_resize_images(pred_folder)

# Make predictions on test set
y_pred = dt_classifier_best.predict(X_test)

# Print Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualization of Predicted Outputs
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(range(1, len(y_pred) + 1), y_pred, edgecolors=(0, 0, 0))
ax.set_xlabel('Image Index')
ax.set_ylabel('Predicted Labels')
ax.set_title('Decision Tree Predicted Outputs')
plt.show()
