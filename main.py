from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import numpy as np
import os
model = keras.models.load_model('model-v2')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded/images'

@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds(filename):
	print(filename)
	test_datagen = ImageDataGenerator(rescale = 1./255)
	vals = ['glass','cardboard','metal','paper','plastic','trash'] # change this according to what you've trained your model to do
	test_dir = 'uploaded'
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size =(150, 150),
			color_mode ="grayscale",
			shuffle = False,
			class_mode ='categorical',
			batch_size = 1)

	pred = model.predict_generator(test_generator)
	print(pred)
	os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	return str(vals[np.argmax(pred)])

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		print('done')
		val = finds(f.filename)
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run()
