# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution Layer
model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
model.add(Convolution2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening Layer
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# you should replace 'dataset/traning_set' and 'dataset/test_set' with you dataset location. I do this scripting through google collab, place the script and folder dataset on the same folder
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
label_map = (training_set.class_indices)
label_map = dict((value,key) for key,value in label_map.items()) #flip key,value

#mapping label to JSON format and save to JSON file
import json

with open('label.json', 'w') as json_file:
  json.dump(label_map, json_file)

model.fit_generator(
    training_set,
    steps_per_epoch=4000,
    epochs=8,
    validation_data=test_set,
    validation_steps=400)

#uncomment this model if you want to try modelling with steps 8000 and epochs 10
"""
model.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=800)
"""


#############
# Export the model can be done by 2 ways:
# 1. separate file : weights into h5 and model into JSON
# 2. whole model into h5 file

from keras.models import model_from_json
import os

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# save model and architecture to single file
model.save("model_full.h5")
print("Saved model to disk")