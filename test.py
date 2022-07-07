#Loading the model
from tensorflow.keras.models import model_from_json
# Model reconstruction from JSON file
with open('model2_0.01_100.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('model2_0.01_100.h5')
#Preprocesisng for text image
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
test_image=tf.keras.utils.load_img('burger.jpg',target_size=(150,150))
test_image=tf.keras.utils.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
test_image=test_image/255
#printing the result
result = model.predict(test_image)
if result[0][0]<=0.5:
    prediction='burger'
else:
    prediction='vadapaav'
print(prediction)