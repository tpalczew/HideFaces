# HideFaces.AI
Author: Tomasz Palczewski

## Motivation for this project:
- “Dealing with privacy, both legal requirements and social norms, is hard but necessary” John Hanke, director of Google Earth and Google Maps

- Extremely hard task that has not been really fully solved so far as even one missed detection may have a huge legal consequences

## Main Goal

- My main goal was to build a pipeline that tests different Deep Learning algorithms available on the market and study their efficiency / misclassification examples for the specific task of face anonymization. In this project, Idecided to implement and test two algorithms: Haar Cascade Classifier and You Only Look Once (YOLO) deep learning single stage detector. 
To produce a minimum viable product quickly, I decide to use previously trained Haar Classifier (trained on frontal face images). The YOLO implementation is based on Keras (high-level neural networks API). I decided to performe transfer learning for YOLO model pretrained on COCO Dataset.     

<p align="center">
  <img src="static/Images-AT.001.jpeg" width="80%" title="Now you see me (left), Now you don't (right)">
</p>


## Setup
- Clone repo
```
git clone https://github.com/tpalczew/HideFaces.git
cd HideFaces
```

## Requisites
- To download all needed requisites
```
pip install -r build/requirements.txt
```
The project was developed and tested on AWS E2 instance using Deep Learning AMI (Ubuntu) Version 15.0 (ami-0b43cec40e1390f34).

The pip freez for this specific setting shows: 

absl-py==0.5.0

alabaster==0.7.10

anaconda-client==1.6.14

anaconda-project==0.8.2

asn1crypto==0.24.0

astor==0.7.1

astroid==1.6.3
astropy==3.0.2
attrs==18.1.0
Automat==0.3.0
autovizwidget==0.12.5
Babel==2.5.3
backcall==0.1.0
backports-abc==0.5
backports.shutil-get-terminal-size==1.0.0
base58==1.0.2
beautifulsoup4==4.6.0
bitarray==0.8.1
bkcharts==0.2
blaze==0.11.3
bleach==2.1.3
bokeh==0.12.13
boto==2.48.0
boto3==1.7.82
botocore==1.10.84
Bottleneck==1.2.1
certifi==2018.8.24
cffi==1.11.5
characteristic==14.3.0
chardet==3.0.4
click==6.7
cloudpickle==0.5.3
clyent==1.2.2
colorama==0.3.9
contextlib2==0.5.5
cryptography==2.2.2
cycler==0.10.0
Cython==0.28.4
cytoolz==0.9.0.1
dask==0.17.5
datashape==0.5.4
decorator==4.3.0
distributed==1.21.8
docutils==0.14
entrypoints==0.2.3
enum-compat==0.0.2
environment-kernels==1.1.1
et-xmlfile==1.0.1
fastcache==1.0.2
filelock==3.0.4
Flask==1.0.2
Flask-Cors==3.0.4
future==0.16.0
future-fstrings==0.4.4
futures==3.1.1
gast==0.2.0
gevent==1.3.0
glob2==0.6
gmpy2==2.0.8
greenlet==0.4.13
grpcio==1.10.1
h5py==2.8.0
hdijupyterutils==0.12.5
heapdict==1.0.0
horovod==0.13.11
html5lib==1.0.1
idna==2.6
imageio==2.3.0
imagesize==1.0.0
imgaug==0.2.6
ipykernel==4.8.2
ipyparallel==6.2.2
ipython==6.4.0
ipython-genutils==0.2.0
ipywidgets==7.4.0
isort==4.3.4
itsdangerous==0.24
jdcal==1.4
jedi==0.12.0
Jinja2==2.10
jmespath==0.9.3
jsonschema==2.6.0
jupyter==1.0.0
jupyter-client==5.2.3
jupyter-console==5.2.0
jupyter-core==4.4.0
jupyterlab==0.32.1
jupyterlab-launcher==0.10.5
Keras==2.2.2
Keras-Applications==1.0.4
Keras-Preprocessing==1.0.2
kiwisolver==1.0.1
lazy-object-proxy==1.3.1
llvmlite==0.23.1
locket==0.2.0
lxml==4.2.1
Markdown==3.0
MarkupSafe==1.0
matplotlib==2.2.2
mccabe==0.6.1
mistune==0.8.3
mkl-fft==1.0.0
mkl-random==1.0.1
mock==2.0.0
more-itertools==4.1.0
mpmath==1.0.0
msgpack==0.5.6
msgpack-python==0.5.6
multipledispatch==0.5.0
nb-conda==2.2.1
nb-conda-kernels==2.1.1
nbconvert==5.3.1
nbformat==4.4.0
networkx==2.1
nltk==3.3
nose==1.3.7
notebook==5.5.0
numba==0.38.0
numexpr==2.6.5
numpy==1.14.5
numpydoc==0.8.0
odo==0.5.1
olefile==0.45.1
opencv-python==3.4.3.18
openpyxl==2.5.3
packaging==17.1
pandas==0.23.4
pandocfilters==1.4.2
parso==0.2.0
partd==0.3.8
path.py==11.0.1
pathlib2==2.3.2
patsy==0.5.0
pbr==4.2.0
pep8==1.7.1
pexpect==4.5.0
pickleshare==0.7.4
Pillow==5.2.0
pkginfo==1.4.2
plotly==2.7.0
pluggy==0.6.0
ply==3.11
prompt-toolkit==1.0.15
protobuf==3.6.0
protobuf3-to-dict==0.1.5
psutil==5.4.7
psycopg2==2.7.5
ptyprocess==0.5.2
py==1.5.3
py4j==0.10.4
pycodestyle==2.4.0
pycosat==0.6.3
pycparser==2.18
pycrypto==2.6.1
pycurl==7.43.0.1
pydot==1.2.4
pyflakes==1.6.0
pygal==2.4.0
Pygments==2.2.0
pykerberos==1.2.1
pylint==1.8.4
pyodbc==4.0.23
pyOpenSSL==18.0.0
pyparsing==2.2.0
PySocks==1.6.8
pyspark==2.2.1
pytest==3.5.1
pytest-arraydiff==0.2
pytest-astropy==0.3.0
pytest-doctestplus==0.1.3
pytest-openfiles==0.3.0
pytest-remotedata==0.2.1
python-dateutil==2.7.3
pytz==2018.4
PyWavelets==0.5.2
PyYAML==3.12
pyzmq==17.0.0
QtAwesome==0.4.4
qtconsole==4.3.1
QtPy==1.4.1
requests==2.18.4
requests-kerberos==0.12.0
rope==0.10.7
ruamel-yaml==0.15.35
s3fs==0.1.5
s3transfer==0.1.13
sagemaker==1.9.0
sagemaker-pyspark==1.1.3
scikit-image==0.13.1
scikit-learn==0.19.1
scipy==1.1.0
seaborn==0.8.1
Send2Trash==1.5.0
simplegeneric==0.8.1
singledispatch==3.4.0.3
six==1.11.0
sklearn==0.0
snowballstemmer==1.2.1
sortedcollections==0.6.1
sortedcontainers==1.5.10
sparkmagic==0.12.5
Sphinx==1.7.4
sphinxcontrib-websupport==1.0.1
spyder==3.2.8
SQLAlchemy==1.2.11
statsmodels==0.9.0
streamlit==0.16.1
sympy==1.1.1
tables==3.4.3
tblib==1.3.2
tensorboard==1.10.0
tensorflow==1.10.0
termcolor==1.1.0
terminado==0.8.1
testpath==0.3.1
tokenize-rt==2.1.0
toolz==0.9.0
tornado==5.0.2
tqdm==4.26.0
traitlets==4.3.2
typing==3.6.4
tzlocal==1.5.1
unicodecsv==0.14.1
urllib3==1.22
wcwidth==0.1.7
webencodings==0.5.1
Werkzeug==0.14.1
widgetsnbextension==3.4.2
wrapt==1.10.11
xlrd==1.1.0
XlsxWriter==1.0.4
xlwt==1.3.0
zict==0.1.3


## Environment Settings
```
source ./build/environment.sh
```


## Test
- All tests are placed in the tests directory. To run tests:
```
cd tests
python tests.py
```
as an output one should see a similar output:
<p align="center">
  <img src="static/test_out.png" width="80%">
</p>



## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```


## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
