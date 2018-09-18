import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

try:
   base_dir = os.environ.get("BASE_DIR")
   print(base_dir)
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


def load_pictures_for_predictions(img, ):
    """ Load pictures for predictions"""
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)




def load_pictures_for_tuning(img, labels):
    """ Load pictures with labels for fine tuning"""
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)
