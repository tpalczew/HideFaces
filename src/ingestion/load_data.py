import os
import cv2
import numpy as np

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


def load_pictures_haar(img):
    """ Load pictures for predictions (Haar)"""
    image = cv2.imread(img)
    img_arr = cv2.resize(image,(224,224))
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_shape = img_arr.shape
    return image, img_arr, gray, img_shape

def load_pictures_yolo(img):
    """ Load pictures for predictions (YOLO)"""
    image = cv2.imread(img)
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    return image, input_image
