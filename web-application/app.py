import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

base_model = VGG19(include_top=False, input_shape=(240, 240, 3))

image_base_model=base_model.output
flat = Flatten()(image_base_model)

first_layer_class = Dense(4608, activation = 'relu')(flat)
drop_out = Dropout(0.3)(first_layer_class)
second_layer_class = Dense(1152, activation = 'relu')(drop_out)
output_layer = Dense(2, activation = 'softmax')(first_layer_class)

image_model_03 = Model(base_model.inputs, output_layer)
image_model_03.load_weights('../cnn-models/Brain Tumor Classification/model_weights/vgg19_image_model_03.weights.h5')
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=image_model_03.predict(input_img)
    result01=np.argmax(result,axis=1)
    return result01


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'predict_image', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)