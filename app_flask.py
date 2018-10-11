import os
import logging
from PIL import Image
from model import FaceCascade, FindFaces, Blur, yolo_v2_model, Blur_yolo
from load_data import load_pictures_haar, load_pictures_yolo
import matplotlib.pyplot as plt
from optparse import OptionParser
import cv2
import numpy as np
from utils_yolo_v2 import WeightReader, decode_netout, draw_boxes
import io
import base64
from flask import Flask, render_template, request, url_for, redirect, send_file
from werkzeug import secure_filename
from keras import backend as K


try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

UPLOAD_FOLDER = str(base_dir) + '/data/flask/'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index')
@app.route('/')
def index():
    return render_template('settings.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """
    model_name = request.form['model']
    blur_mode = request.form['blur']

    #
    if model_name in ['haar', 'yolo']:
        architecture = model_name
    else:
        architecture = 'haar'
    K.clear_session()
    if model_name == 'yolo':
        model, wt_path, model_png_path  = yolo_v2_model()
    if model_name == 'haar':
        model, dir_haar_face = FaceCascade()
    if request.method == 'POST' and 'file[]' in request.files:
        inputs = []
        files = request.files.getlist('file[]')
        outputs = []
        for file_obj in files:
            """ At this stage I am focus only on one file to show a proof of concept,
                In future this part might be changed to handle a few pictures at the
                same time
            """
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue
            if file_obj.filename != '':
                if len(files) >= 2:
                    return render_template('one_file_only.html')
            filename = secure_filename(file_obj.filename)
            imgdir = str(base_dir) + '/data/flask/'
            filepath = os.path.join(imgdir, filename)
            file_obj.save(filepath)
            if architecture == 'haar':
                haar_filename = 'blur_haar_' + str(filename)
                outPath = str(base_dir) + '/data/flask/' + str(haar_filename)
                try:
                    img, img_arr, gray, img_shape = load_pictures_haar(filepath)
                except cv2.error as e:
                    raise
                faces = FindFaces(gray, model)
                final_picture = Blur(img_arr, faces, blur_mode)
                outputs.append(haar_filename)
                final_picture = cv2.cvtColor(final_picture, cv2.COLOR_RGB2BGR)
                cv2.imwrite(outPath, final_picture)
                K.clear_session()
                return send_file(outPath, mimetype='image/jpg')
            if architecture == 'yolo':
                yolo_filename = 'blur_yolo_' + str(filename)
                outPath = str(base_dir) + '/data/flask/' + str(yolo_filename)
                try:
                    image, input_image = load_pictures_yolo(filepath)
                except cv2.error as e:
                    raise
                dummy_array = np.zeros((1,1,1,1,50,4))
                netout = model.predict([input_image, dummy_array])
                boxes = decode_netout(netout[0],
                                  obj_threshold=0.3,
                                  nms_threshold=0.3,
                                  anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                                  nb_class=1)
                LABELS = ['face']
                image = Blur_yolo(image, boxes, blur_mode, labels=LABELS)
                outputs.append(yolo_filename)
                cv2.imwrite(outPath, image)
                K.clear_session()
                return send_file(outPath, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)
