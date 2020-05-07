from numpy import loadtxt
from keras.models import load_model

model = load_model('model.h5')

CATEGORIES = ["Grass", "Dirt", "Building", "Road", "Trees"]

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('1588404587 3,0.png', target_size=(50,50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict([test_image])


print(CATEGORIES[int(result)])

