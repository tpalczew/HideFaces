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
The project was developed and tested on AWS E2 instance using Deep Learning AMI (Ubuntu) Version 15.0 (ami-0b43cec40e1390f34)
using python3.

The pip freez for this specific setting is shown [here](https://github.com/tpalczew/HideFaces/blob/master/static/aws-e2-requirements.txt)


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
Face detection and blurring on a single image may be performed using command line script
```
python app.py --infile /dir/subdir/file.jpg --outfile /dir/subdir/file_blur.jpg --blur blur_type --model model_type
```
available blur types are as follows: Median Blur (use medianBlur), Gaussian Blur (use GaussianBlur), Bilateral Filter (use bilateralFilter), Averaging (use blur). The default blur is a Median Blur. 

there are two models that can be used to detect faces: Haar Cascade Classifier (use haar) or YOLO (use yolov2).
The default model is a Haar Cascade Classifier


## Serve Model

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
