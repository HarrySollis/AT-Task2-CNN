from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation, Dense
import os

dirs = os.listdir('TrainingData/Satellite/TrainingFolders/')

classifier = Sequential()


classifier.add(Convolution2D(64,(3,3),input_shape = (50,50,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=32))
classifier.add(Dense(activation="softmax", units=5))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    shear_range = 0.2, 
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1/255)

training_set =  train_datagen.flow_from_directory(
    'TrainingData/Satellite/TrainingFolders',
    target_size = (50,50),
    batch_size = 64,
    color_mode='rgb',
    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
    'TrainingData/Satellite/TestFolders',
    target_size=(50,50),
    batch_size=64,
    color_mode='rgb',
    class_mode='categorical')

print(test_set.class_indices)

from IPython.display import display
from PIL import Image

classifier.fit(
    training_set,
    steps_per_epoch=100,
    epochs=1,
    validation_data=test_set,
    validation_steps=1500)

classifier.save("model.h5")
print("Saved Model to Disk")