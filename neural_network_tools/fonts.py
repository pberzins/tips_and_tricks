
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.regularizers import l2
import tensorflow as tf
from PIL import Image

from src.character_images import CharacterImages

def make_a_model(dropout_rate=0.5,hidden_nodes=512):
    model= Sequential()

    #Layer1
    model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(1,1),
                    activation='relu', input_shape=(32,32,1),
                    kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


    #Layer2
    model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(1,1),
                    activation= 'relu', input_shape=(32,32,1),
                    kernel_regularizer= l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


    #Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(hidden_nodes, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))

    #Output Layer
    model.add(Dense(4, activation='softmax', kernel_regularizer=l2(0.01)))

    #Compiler
    model.compile(loss = keras.losses.categorical_crossentropy,
                 optimizer= keras.optimizers.Adam(lr=0.001),
                 metrics= ['accuracy'])
    return model

if __name__ == "__main__":
    tnr = 'Times New Roman'
    comic= 'Comic Sans MS'
    courier= 'Courier New'
    arial= 'Arial Black'
    fonts_list = [tnr,comic,courier,arial]

    fonts = CharacterImages(fonts_list)
    fonts.generate_char_files()
    test = fonts.load_images('test')
    train = fonts.load_images('train')

    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]

    X_train=X_train.reshape(13248,32, 32)
    X_test = X_test.reshape(4608, 32, 32)

    model = make_a_model(0.5,hidden_nodes=512)

    model.fit(X_train[...,None], y_train,
         epochs = 10, verbose = 1,
         batch_size= 50,
         validation_data=(X_test[...,None],y_test))
