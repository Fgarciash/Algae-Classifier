#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
import tensorflow-cpu
	
from tensorflow-cpu.python.keras.optimizers import Adam

from tensorflow-cpu.keras.models import model_from_json
from tensorflow-cpu.python.framework import ops
ops.reset_default_graph()
from tensorflow-cpu.python.keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)


# ::: Prepare Keras Model :::
# Model files

MODEL_ARCHITECTURE = 'models/model_adam_20210613_01.json'
MODEL_WEIGHTS = 'models/model_5_eopchs_adam_20210613_01.h5'


# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
#print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path).convert('RGB')
	print(type(IMG))

	# Pre-processing the image
	IMG_ = IMG.resize((224, 224))
#	IMG = np.expand_dims(IMG,axis=0)
	print(type(IMG_))
	IMG_ = np.asarray(IMG_)
	print(IMG_.shape)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = IMG_.reshape(1, 224, 224, 3)
	print(type(IMG_), IMG_.shape)

	print(model)

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	predict2 = model.predict(IMG_)
#	prediction = model.predict_classes(IMG_)
	prediction = predict2.argmax(axis=-1)

	return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'ESPECIE': ['especie1', 'especie2', 'especie3', 'especie4', 'especie5', 'especie6', 'especie7', 'especie8', 'especie9', 'especie10']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)

		predicted_class = classes['ESPECIE'][prediction[0]]
		print('El alga es {}.'.format(predicted_class.lower()))

		return str(predicted_class).lower()
		

if __name__ == '__main__':
	app.run(debug = True) 