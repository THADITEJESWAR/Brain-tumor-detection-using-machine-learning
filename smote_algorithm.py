import os
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE

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

# Example usage
yes_folder = r'D:\brainTumor\datasets\yes'
no_folder = r'D:\brainTumor\datasets\no'

# Load and print preprocessed data for the 'yes' folder
images_yes, labels_yes = load_and_resize_images(yes_folder)

# Load and print preprocessed data for the 'no' folder
images_no, labels_no = load_and_resize_images(no_folder)

# Combine data for training
X_train = np.concatenate((images_yes, images_no), axis=0)
y_train = np.concatenate((labels_yes, labels_no))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the resampled data
print("\nResampled Data:")
for i, (img_array, label) in enumerate(zip(X_resampled, y_resampled)):
    print(f"Resampled Image {i+1}: {img_array}, Label: {label}")
