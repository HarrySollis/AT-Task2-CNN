import os
from cv2 import cv2
import numpy as np
import requests
import math
import random
from pathlib import Path
import time
from PIL import Image

LAT = [45.35051, 45.65]
LONG = [-122.20, -123.0]

X_SLICE = 50
Y_SLICE = 50

RAW_IMG_PATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\RawImg"
CROPPED_IMG_PATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\Cropped"
SLICED_IMG_PATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\SlicedImg"

def mapquest_image_request():
    randLat = random.uniform(LAT[0], LAT[1])
    randLon = random.uniform(LONG[0], LONG[1])
    url = f"https://www.mapquestapi.com/staticmap/v4/getmap?key=MS6YUJYIiHcFHDwp0L8iUDxsuSw0G7c7&size=400, 400&zoom=18&center={randLat},{randLon}&type=sat&imagetype=png&pois=1, {randLat}, {randLat}, {randLon}, {randLon}"
    response = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #cv2.imshow('URL IMG', image)
    #cv2.waitKey()
    return image
    #return cv2.imdecode(image, cv2.IMREAD_COLOR)

def crop_img():
    raw_img = mapquest_image_request()
    image = raw_img[25:375, 0:400]
    #image = cv2.resize(raw_img, dsize = size, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("croppped", image)
    #cv2.waitKey(0)
    return image

def save_img():
    image = crop_img()
    file_name = f"{int(time.time())}.{'png'}"
    cv2.imwrite(CROPPED_IMG_PATH + "\\" + file_name, image)

for x in range(1):
    save_img()
    
X_SLICE = int(X_SLICE)
Y_SLICE = int(Y_SLICE)

def slice_image(img):
    img_width, img_height = img.shape[:2]
    x_tiles = math.floor(img_width / X_SLICE)
    y_tiles = math.floor(img_height / Y_SLICE)
    image_start_time = f"{int(time.time())}"
    for y in range(y_tiles):
        for x in range(x_tiles):
            current_x_step = x * X_SLICE
            current_y_step = y * Y_SLICE
            new_img = img[current_y_step:current_y_step+Y_SLICE, current_x_step:current_x_step+X_SLICE]
            file_name = f"\\{image_start_time} {x},{y}.png"
            cv2.imwrite(SLICED_IMG_PATH + file_name, new_img)  

for img in os.listdir(CROPPED_IMG_PATH):
    img_array = cv2.imread(os.path.join(CROPPED_IMG_PATH, img), cv2.IMREAD_COLOR)
    slice_image(img_array)


