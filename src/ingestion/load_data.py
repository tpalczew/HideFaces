import os
import cv2

try:
   base_dir = os.environ.get("BASE_DIR")
   print(base_dir)
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


def load_pictures(img, LABELS????):
    """ Load pictures """
    img = img
    img_array = cv2.imread(img, cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array,(224,224))   # what about bounding boxes ???
    rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_shape = rgb_image.shape
    return img_array, rgb_image, gray, img_shape
