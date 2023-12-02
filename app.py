import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    
    # Use predict instead of predict_classes
    raw_predictions = model.predict(input_img)
    
    # Assuming the model has a sigmoid activation in the output layer for binary classification
    predicted_class = int(raw_predictions[0][0] > 0.5)  # Adjust the threshold if needed
    
    return predicted_class


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
