import os
import logging
from PIL import Image
from model import FaceCascade, FindFaces, Blur, HaarBoxes, yolo_v2_model
from load_data import load_pictures
import matplotlib.pyplot as plt
from optparse import OptionParser
import cv2
import numpy as np
from utils_yolo_v2 import WeightReader, decode_netout, draw_boxes
import glob
import time
import xml.etree.ElementTree as ET


try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

blur_mode = 'medianBlur'

dir_jpg_all_files = base_dir + '/data/raw/face/test/*'
all_test_jpg_files = glob.glob(dir_jpg_all_files)


dir_ann_all_files = base_dir + '/data/raw/face/test_ann/*'
all_test_ann_files = glob.glob(dir_ann_all_files)


####### define IoU






############################################

name = []
boxes_true = []
boxes_yolo = []
boxes_haar = []
time_yolo = []
time_haar = []


############################################

haar_model = FaceCascade()

yolo_model = yolo_v2_model()


time_yolo_all = 0
time_haar_all = 0


i = 0
for item_ann in all_test_ann_files:
    i += 1
    print(str(i) + " / " + str(len(all_test_jpg_files)))
    print(item_ann)

    name.append(item_ann)

    ##### parse true xml file #####

    tree = ET.parse(item_ann)


    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            box = []
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            box.append(int(round(float(dim.text))))
                        if 'ymin' in dim.tag:
                            box.append(int(round(float(dim.text))))
                        if 'xmax' in dim.tag:
                            box.append(int(round(float(dim.text))))
                        if 'ymax' in dim.tag:
                            box.append(int(round(float(dim.text))))
            boxes_true.append(box)
    print(boxes_true)
    print(len(boxes_true))

    print(item_ann)
    item = item_ann.replace( "xml", "jpg" )
    item = item.replace("test_ann", "test")
    print(item)

    imagePath = item
    img, img_arr, gray, img_shape = load_pictures(imagePath)

    try:
        img.shape
        print("checked for shape".format(img.shape))
    except AttributeError:
        print("shape not found")

    print("Haar Cascade")

    start_time = time.clock()
    haar_faces = FindFaces(gray, haar_model)
    execution_time_haar = time.clock() - start_time

    time_haar_all += execution_time_haar


    print(haar_faces)
    print(len(haar_faces))

    time_haar.append(execution_time_haar)
    print("Haar execution time = " + str(execution_time_haar))

    final_picture = HaarBoxes(img_arr, haar_faces)

    outPath = item.replace(".jpg", "_blur_Haar.jpg")
    outPath = outPath.replace("test", "tested")

    print(outPath)
    plt.imshow(final_picture)
    plt.savefig(outPath)

    print("YOLO")
    image = cv2.imread(imagePath)
    dummy_array = np.zeros((1,1,1,1,50,4))

    plt.figure(figsize=(10,10))

    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)


    start_time = time.clock()
    yolo_netout = yolo_model.predict([input_image, dummy_array])
    execution_time_yolo = time.clock() - start_time
    time_yolo.append(execution_time_yolo)

    time_yolo_all += execution_time_yolo

    print("YOLO execution time = " + str(execution_time_yolo))

    boxes = decode_netout(yolo_netout[0],
                      obj_threshold=0.3,
                      nms_threshold=0.3,
                      anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                      nb_class=1)

    yolo_faces = []
    for box in boxes:
        image_h, image_w, _ = image.shape
        x = int(box.xmin*image_w)
        w = int(box.ymin*image_h)
        y = int(box.xmax*image_w)
        h = int(box.ymax*image_h)
        yolo_faces.append([x, y, w, h])
    print(yolo_faces)
    print(len(yolo_faces))

    LABELS = ['face']
    image_with_boxes = draw_boxes(image, boxes, labels=LABELS)

    #final_picture = Blur(image, faces, blur_mode)

    outPath = item.replace(".jpg", "_blur_YOLO.jpg")
    outPath = outPath.replace("test", "tested")
    print(outPath)

    plt.imshow(image_with_boxes[:,:,::-1]);
    #plt.imshow(final_picture)
    plt.savefig(outPath)


print(name)
print(boxes_true)
print(boxes_yolo)
print(boxes_haar)
print(time_yolo)
print(time_haar)


print(" ")
print(" ")
print("Haar Time total")
print(time_haar_all)
print("YOLO Time total")
print(time_yolo_all)
