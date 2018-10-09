import os
import logging
from flask import Flask, render_template, request
from PIL import Image
from model import FaceCascade, FindFaces, Blur
from load_data import load_pictures
import matplotlib.pyplot as plt
from optparse import OptionParser
import io
import base64

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


parser = OptionParser()
parser.allow_interspersed_args = True
parser.add_option("-i", "--infile",default="/Users/tpalczew/SomePhotos/DSC_007.JPG",
                  dest="INFILE", help="input picture in JPG format (full path)")
parser.add_option("-o", "--outfile",default="/Users/tpalczew/SomePhotos/DSC_007_blur.JPG",
                  dest="OUTFILE", help="output picture (full path)")
parser.add_option("-b", "--blur", default="medianBlur",
                  dest="BLUR", help="blur type (medianBlur, GaussianBlur, bilateralFilter, blur)")
parser.add_option("-m", "--model", default="CascadeClassifier",
                  dest="MODEL", help="detection model (CascadeClassifier, )")


(options,args) = parser.parse_args()

imagePath = options.INFILE
outPath = options.OUTFILE
blurMode = options.BLUR

logger = logging.getLogger("Hide Faces")
logger.setLevel(logging.DEBUG)


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
            inputs.append(entry)

        outputs = []

        with graph.as_default():
            for input_ in inputs:
                # convert to 4D tensor to feed into our model
                x = preprocess(input_['data'])
                # perform prediction
                out = model.predict(x)
                outputs.append(out)

        # decode output prob
        outputs = decode_prob(outputs)

        results = []

        for input_, probs in zip(inputs, outputs):
            results.append({'filename': input_['filename'],
                            'image': input_['img'],
                            'predict_probs': probs})

        return render_template('results.html', results=results)

    # if no files uploaded
    return render_template('select_files.html')


if __name__ == '__main__':
    app.run(debug=True)
