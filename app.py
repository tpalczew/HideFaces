import os
import logging
from PIL import Image
from model import FaceCascade, FindFaces, Blur, yolo_v2_model
from load_data import load_pictures_haar, load_pictures_yolo
import matplotlib.pyplot as plt
from optparse import OptionParser
import cv2
import numpy as np
from utils_yolo_v2 import WeightReader, decode_netout, draw_boxes


logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("logger-app.py")


parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile",default="/home/ubuntu/Insight/data/raw/face/test/18_Concerts_Concerts_18_1009.jpg",
                  dest="INFILE", help="input picture in JPG format (full path)")
parser.add_option("-o", "--outfile",default="/home/ubuntu/Insight/data/raw/face/18_Concerts_Concerts_18_1009_blur_Haar.jpg",
                  dest="OUTFILE", help="output picture (full path)")
parser.add_option("-b", "--blur", default="medianBlur",
                  dest="BLUR", help="blur type (medianBlur, GaussianBlur, bilateralFilter, blur)")
parser.add_option("-m", "--model", default="CascadeClassifier",
                  dest="MODEL", help="detection model (CascadeClassifier)")

(options,args) = parser.parse_args()
imagePath = options.INFILE
outPath = options.OUTFILE
blurMode = options.BLUR
Architecture = options.MODEL

logger.info('HideFaces.AI')
logger.info('Options:')
logger.info('--infile = %s', imagePath)
logger.info('--outfile = %s', outPath)


if blurMode in ['medianBlur', 'GaussianBlur', 'bilateralFilter', 'blur']:
    blur_mode = blurMode
    logger.info('--blur = %s', blurMode)
else:
    blur_mode = 'medianBlur'
    logger.info('--blur = %s', blurMode)

if Architecture in ['haar', 'yolov2']:
    architecture = Architecture
    logger.info('--model = %s', Architecture)
else:
    architecture = 'haar'
    logger.info('--model = %s', Architecture)


if architecture == 'haar':
    try:
        img, img_arr, gray, img_shape = load_pictures_haar(imagePath)
        logger.info('picture loaded')
    except cv2.error as e:
        logger.error('Cannot load pictures from = %s; error = %s', imagePath, e)
        raise

    model, dir_haar_face = FaceCascade()
    logger.info('Haar Cascade Classifier loaded with weights %s', dir_haar_face)

    faces = FindFaces(gray, model)
    logger.info('Haar Cascade Classifier found %s faces', len(faces))

    final_picture = Blur(img_arr, faces, blur_mode)
    logger.info('Face blurring finished')
    plt.imshow(final_picture)
    plt.savefig(outPath)

if architecture == 'yolov2':
    try:
        image, input_image = load_pictures_yolo(imagePath)
    except cv2.error as e:
        logger.error('Cannot load pictures from = %s; error = %s', imagePath, e)
        raise

    model, wt_path, model_png_path  = yolo_v2_model()
    logger.info('Yolo model loaded')
    logger.info('Yolo weights from %s are used', wt_path)
    logger.info('Yolo model stored as png in %s', model_png_path)

    dummy_array = np.zeros((1,1,1,1,50,4))

    plt.figure(figsize=(10,10))
    netout = model.predict([input_image, dummy_array])
    boxes = decode_netout(netout[0],
                      obj_threshold=0.3,
                      nms_threshold=0.3,
                      anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                      nb_class=1)
    logger.info('Yolo found %s faces', len(boxes))
    LABELS = ['face']
    image = draw_boxes(image, boxes, labels=LABELS)
    logger.info('Face blurring finished')
    plt.imshow(image[:,:,::-1]);
    plt.savefig(outPath)
