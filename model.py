import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# %%
p_train = "./train"
p_val = "./val"
p_test = "./test"

# %%
img = glob(p_train+"/PNEUMONIA/*.jpeg")

# %%
classes = ["NORMAL", "PNEUMONIA"]
train_data = glob(p_train+"/NORMAL/*.jpeg")
train_data += glob(p_train+"/PNEUMONIA/*.jpeg")
data_gen = ImageDataGenerator()
train_batches = data_gen.flow_from_directory(p_train, target_size=(226, 226), classes=classes, class_mode="categorical")
val_batches = data_gen.flow_from_directory(p_val, target_size=(226, 226), classes=classes, class_mode="categorical")
test_batches = data_gen.flow_from_directory(p_test, target_size=(226, 226), classes=classes, class_mode="categorical")


# %%
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_batches.image_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# %%
model.summary()

# %%
optimizer = Adam(learning_rate=0.0001)
early_stopping = EarlyStopping(patience=5, min_delta=0.001, monitor="val_acc", mode="max", verbose=2)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit(train_batches, epochs=5, callbacks=[early_stopping], shuffle=True, validation_data=val_batches, steps_per_epoch=500, validation_steps=10, verbose=2)
history.save("new69.h5")

