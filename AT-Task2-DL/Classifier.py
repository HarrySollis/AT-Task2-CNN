from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import TensorBoard
import time
import os

dirs = os.listdir('TrainingData/Satellite/TrainingFolders/')

name = "Terrain_Classifier-CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\{}'.format(name))

imgSize = 100

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (imgSize, imgSize, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128))
classifier.add(ac)