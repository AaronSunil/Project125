import os, ssl, time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import PIL.ImageOps
from sklearn.datasets import fetch_openml
import cv2
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression

#Loading the image.npz file and labels .csv file to X and Y respectively
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

#Making the classes and nclasses
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

#Spliting the data into Training And Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#making the classifier with Logistic Regression
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    #Opening the image
    im_pil = Image.open(image)

    #Converting and Resizing the image
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS)

    #Applying the pixel fileter and using the np percentile function
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    #USing the np.clip,max aand asarray functions 
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    #Using the array and reshape functions to make the test sample
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    #Using the predict function to make the test prediction
    test_pred = clf.predict(test_sample)
    #Returning the test prediction variable
    return test_pred[0]