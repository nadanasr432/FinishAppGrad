import tensorflow
from keras.applications.vgg19 import VGG19
from IPython.display import clear_output
from keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout
from sklearn.metrics import *
from sklearn.model_selection import *
import keras.backend as K
from tqdm import tqdm, tqdm_notebook
from colorama import Fore
import json
import matplotlib.pyplot as plt
import seaborn as sns
#from glob import globcd
from skimage.io import *
#%config Completer.use_jedi = False
import time
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import sys
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import joblib
init_notebook_mode(connected=True)
RANDOM_SEED = 123
IMG_SIZE = (224,224)
def image_crop(image):
    img = cv2.resize(
        image,
        dsize=IMG_SIZE,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # add contour on the image
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # crop
    ADD_PIXELS = 0
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
    return new_img


#### predict tumer

def predict_tumer(image_path):
    set = []
    image = cv2.imread(image_path)
    image = image_crop(image)
    image = cv2.resize(
        image,
        dsize=(224, 224),
        interpolation=cv2.INTER_CUBIC
    )
    image = preprocess_input(image)
    set.append(image)
    set = np.array(set)
    model=joblib.load('model.pkl')
    predict_img = model.predict(set)
    predict_img = [1 if x > 0.5 else 0 for x in predict_img]

    threshold = 0.5
    if predict_img[0] > threshold:
        result = "Tumer detected."
    else:
        result = "No tumer detected."
    return result


# Get the path to the image file from the command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py /path/to/image")
    sys.exit()

image_path = sys.argv[1]

# Check if the file exists
if not os.path.exists(image_path):
    print("File not found:", image_path)
    sys.exit()

print(predict_tumer(image_path))