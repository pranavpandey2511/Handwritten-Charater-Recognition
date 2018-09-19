import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
import cv2

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
                cv2.circle(img,(x,y),10,(0,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),10,(0,0,0),-1) 


img = np.ones((512,512,3))
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(0) #Waitkey
cv2.destroyAllWindows()

train = pd.read_csv('E:\\datasets\\emnist\\emnist-letters-train.csv')
test = pd.read_csv('E:\\datasets\\emnist\\emnist-letters-test.csv')
train.info()
print(train.head())
Y_train = train['23']


X_train = train.drop(labels = ["23"],axis = 1)

del train

g = sns.countplot(Y_train)

plt.show()

print(Y_train.value_counts())