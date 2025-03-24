import numpy as np
import json
from glob import glob
import pickle,cv2,glob,os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
from random import randint
import base64
from datetime import datetime
import models as dbHandler
import sqlite3
# import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.backend import set_session, get_session
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Lambda, ELU, Cropping2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


dog_names = []
with open('dogs_name.json') as json_file:
    dog_names = json.load(json_file)


ResNet50_model_for_dog_breed = ResNet50(weights='imagenet')


bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet = bottleneck_features['train']
valid_Resnet = bottleneck_features['valid']
test_Resnet = bottleneck_features['test']
Resnet_Model = Sequential()
Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
Resnet_Model.add(Dense(128, activation='relu'))
Resnet_Model.add(Dropout(0.5))
Resnet_Model.add(Dense(133, activation='softmax'))
Resnet_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')

def capture_image():
    cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
    ret, frame = cap.read()    # Capture frame-by-frame
    cap.release()              # Release the camera
    return frame

def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False,pooling="avg").predict(preprocess_input(tensor))

# define generic function for pre-processing images into 4d tensor as input for CNN
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# predicts the dog breed based on the pretrained ResNet50 models with weights from imagenet
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_for_dog_breed.predict(img))

# returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    
    prediction = ResNet50_predict_labels(img_path)
    print(prediction)
    return ((prediction <= 268) & (prediction >= 151))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    print(len(faces))
    return len(faces) > 0


def Resnet_predict_breed(img_path):
    # extract bottleneck features
    #bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    print(bottleneck_feature.shape) #returns (1, 2048)
    # bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    # bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    # bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    #obtain predicted vector
    predicted_vector = Resnet_Model.predict(np.expand_dims(np.expand_dims(bottleneck_feature, axis=0), axis=0))
    # predicted_vector = Resnet_Model.predict(bottleneck_feature) #shape error occurs here
    #return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]



def predict_image(img_path):
    """
    This function displays the type of an image breed    
    Args:
        img_path: Image path    
    Returns:
        Str: Print the type of breed of an image 
    """            
    # if a dog is detected in the image, return the predicted breed.

    if dog_detector(img_path) == True:
        print("It's Dog")
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[
            1].replace("_", " ")

        return "The predicted dog breed is " + str(predicted_breed) + "."
    # if a human is detected in the image, return the resembling dog breed.
    if face_detector(img_path) == True:
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[
            1].replace("_", " ")
        # print("Hey, It's a human!")
        return "Hey, It's a human!"
    # if neither is detected in the image, provide output that indicates an error.
    else:
        return "No human or dog could be detected, please provide another picture."
