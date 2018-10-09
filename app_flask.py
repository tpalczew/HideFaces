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
import io
import base64

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")



app = Flask(__name__)

@app.route('/index')
@app.route('/')
def index():
    return render_template('settings.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """
    model_name = request.form['model']
    blur_name = request.form['blur']
    #
    if request.method == 'POST' and 'file[]' in request.files:
        inputs = []
        files = request.files.getlist('file[]')
        for file_obj in files:
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue
            entry = {}
            entry.update({'filename': file_obj.filename})
            try:
                img_bytes = io.BytesIO(file_obj.stream.getvalue())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            except AttributeError:
                img_bytes = io.BytesIO(file_obj.stream.read())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            entry.update({'img': img_b64})

        outputs = []

        with graph.as_default():
            if model_name == 'haar':
                model, dir_haar_face = FaceCascade()
            if model_name == 'yolo':
                model, wt_path, model_png_path  = yolo_v2_model()

            for input_ in inputs:

                #------- Haar
                if model_name == 'haar':
                    try:
                        img, img_arr, gray, img_shape = load_pictures_haar(imagePath)
                    except cv2.error as e:
                        raise

                    faces = FindFaces(gray, model)
                    final_picture = Blur(img_arr, faces, blur_mode)

                #------- Yolo
                if model_name == 'yolov2':
                    try:
                        image, input_image = load_pictures_yolo(imagePath)
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
                    image = draw_boxes(image, boxes, labels=LABELS)

        return render_template('results.html', results=results)

    # if no files uploaded
    return render_template('select_files.html')


if __name__ == '__main__':
    app.run(debug=True)
