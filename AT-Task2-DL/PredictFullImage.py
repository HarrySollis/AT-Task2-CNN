from numpy import loadtxt
from keras.models import load_model
from cv2 import cv2
import os
import math
from PIL import Image
from os import listdir
from os.path import  isfile, join
from keras.preprocessing.image import ImageDataGenerator

FILEPATH = 'C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\SlicedImg\\'
END_FILEPATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\TrainingFolders\\Building\\"

model = load_model('model3.h5')

CATEGORIES = ["Building", "Dirt", "Grass", "Road", "Trees"]

onlyfiles = [f for f in listdir(FILEPATH) if isfile(join(FILEPATH, f))]

import numpy as np
from keras.preprocessing import image
file = open("Level.txt","a+")

for img_path in onlyfiles:
    #_image = Image.open(FILEPATH + img_path)
    test_image = image.load_img(FILEPATH + img_path, target_size=(50,50))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    result = model.predict(test_image)
    
    #print(result)

    output = ""
    for i in range (len(result)):
        output+=(CATEGORIES[result[i].tolist().index(max(result[i]))])
        
    #print(output)
    file.write("\n" + output)

file.close()

