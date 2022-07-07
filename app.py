from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
# Model reconstruction from JSON file
with open('model2_0.01_100.json', 'r') as f:
	model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('model2_0.01_100.h5')
#Preprocesisng for text image
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

def predict_label(img_path):
	test_image=tf.keras.utils.load_img('burger.jpg',target_size=(150,150))
	test_image=tf.keras.utils.img_to_array(test_image)
	test_image=np.expand_dims(test_image,axis=0)
	test_image=test_image/255
	#printing the result
	result = model.predict(test_image)
	if result[0][0]<=0.5:
		return 'burger'
	else:
		return 'vadapaav'

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static\\" + img.filename		
		img.save(img_path)
		p = predict_label(img_path)
		print(p)
	return render_template("home.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)