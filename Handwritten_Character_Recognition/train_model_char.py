import cv2
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import model_from_json





#create CNN model
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
  
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
  
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
  
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(27, activation='softmax'))
  
    return model


def main():
    X_train = pd.read_csv('E:\datasets\EMNIST\emnist-letters-train.csv')
    X_test = pd.read_csv('E:\datasets\EMNIST\emnist-letters-test.csv')
    y_train = X_train['23']
    y_test = X_test['1']

    X_train = X_train.drop(labels = ["23"],axis = 1)
    X_test = X_test.drop(labels = ["1"],axis = 1)

    X_train.shape
    X_test.shape
    y_train.shape
    y_test.shape


    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_train = y_train.reshape(88799,1)
    y_test = np.array(y_test)
    

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_train.shape

    X_train = X_train.reshape(88799,28,28,1)
    X_test = X_test.reshape(14799,28,28,1)
    model1 = createModel()
    batch_size = 1024
    epochs = 20
    model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    save_model(model1)



def save_model(mod):
    model_json = mod.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mod.save_weights("model.h5")
    print("Saved model to disk")




if __name__== '__main__':
    main()