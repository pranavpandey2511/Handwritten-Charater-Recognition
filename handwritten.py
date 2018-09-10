import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')


train = pd.read_csv('emnist-byclass-train.csv')
test = pd.read_csv('emnist-byclass-test.csv')