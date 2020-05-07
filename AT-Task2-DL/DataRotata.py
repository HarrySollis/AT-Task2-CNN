from cv2 import cv2
import os
import math
import time
from PIL import Image
from os import listdir
from os.path import isfile, join

FILEPATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\TrainingFolders\\Building\\"
END_FILEPATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\TrainingFolders\\Building\\"

os.chdir(FILEPATH)

onlyfiles = [f for f in listdir(FILEPATH) if isfile(join(FILEPATH, f))]

image_start_time = f"{int(time.time())}"

def rotateImg(rotVal):
    for img_path in onlyfiles:
        img = Image.open(FILEPATH + img_path)
        rotated_img = img.rotate(rotVal)
        file_name = img_path + "_90.png"
        rotated_img.save(END_FILEPATH + file_name)
        

#print ("CWD = " + dirs)

rotateImg(90)


#for img in os.listdir(FILEPATH):
#    img_array = cv2.imread(os.path.join(FILEPATH, img), cv2.IMREAD_COLOR)
#    rotateImg(img_array)