from cv2 import cv2
import os
import math
import time
from PIL import Image

X_SLICE = 50
Y_SLICE = 50
FILEPATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\RawImg"
END_FILEPATH = "C:\\Users\\harry\\Anaconda3\\envs\\gputest\\AT-Task2-DL\\TrainingData\\Satellite\\SlicedImg"

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
            cv2.imwrite(END_FILEPATH + file_name, new_img)
            

for img in os.listdir(FILEPATH):
    img_array = cv2.imread(os.path.join(FILEPATH, img), cv2.IMREAD_COLOR)
    slice_image(img_array)