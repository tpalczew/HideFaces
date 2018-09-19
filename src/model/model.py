import cv2
import numpy as np


try:
   base_dir = os.environ.get("BASE_DIR")
   print(base_dir)
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")


def FaceCascade():
    """ Pre-trained Haar Cascade model for frontal face detection """
    dir_face_cascade = str(base_dir) + "/preprocessed/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(dir_face_cascade)
    return faceCascade


def FindFaces(gray, model?????):
    """ Find faces and retrun bounding boxes """
    gray = gray
    if model == ????:
        faces = FaceCascade()
        faces.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=15,
            minSize=(5, 5)
            )
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    return faces

def Blur(image, faces):
    image = image
    faces = faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
        bounded_face = image[y:y+h, x:x+w]

        #bounded_face = cv2.GaussianBlur(bounded_face,(23, 23), 30)
        bounded_face = cv2.medianBlur(bounded_face, 55)

        image[y:y+bounded_face.shape[0], x:x+bounded_face.shape[1]] = bounded_face
        x=0
        y=0
        w=0
        h=0
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image
