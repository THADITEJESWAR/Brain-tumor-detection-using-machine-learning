import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer

def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_resized = img.resize(target_size)  # Resize to a common size
        img_array = np.array(img_resized).flatten()  # Flatten the image
        images.append(img_array)
        labels.append("yes" if "yes" in folder else "no")

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

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_train_one_hot = label_binarizer.fit_transform(labels_yes + labels_no)

# Print the one-hot encoded labels
print("One-Hot Encoded Labels:")
print(y_train_one_hot)
