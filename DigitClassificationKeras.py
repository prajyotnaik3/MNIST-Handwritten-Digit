import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


#Load data
(trainX, trainY), (testX, testY) = mnist.load_data()

#Preprocessing
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# convert from integers to floats
trainX_norm = trainX.astype('float32')
testX_norm = testX.astype('float32')
# normalize to range 0-1
trainX_norm = trainX_norm / 255.0
testX_norm = testX_norm / 255.0

earlystopping = EarlyStopping('val_loss', patience = 15, verbose = 1)
modelcheckpoint = ModelCheckpoint('model_uniform.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

#Classification model 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer="SGD", loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainX_norm, trainY, epochs=500, batch_size=32, validation_data=(testX_norm, testY), verbose=1, callbacks = [earlystopping, modelcheckpoint])

# evaluate model
model = load_model('model_uniform.h5')
_, acc = model.evaluate(testX_norm, testY, verbose=1)

