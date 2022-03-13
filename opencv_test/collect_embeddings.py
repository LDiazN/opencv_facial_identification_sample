"""
    Use this script to collect a set of embeddings in a single file 
"""
import cv2
import face_recognition
import numpy
from typing import Any, Tuple

def drawl_rect_label(cv2_image : Any, x : int, y : int, w : int, h : int, label : str, color : Tuple[int, int, int] = (0,255,0)):
    """
        Try to draw a rectangle with a label in an image
    """
    image = cv2.rectangle(cv2_image, (x,y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



cascPathfile = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathfile)

print("Starting capture")
video_capture = cv2.VideoCapture(0)
stored_encodings = []
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in inputP
    encodings = face_recognition.face_encodings(rgb, faces)

    # show faces
    if faces != ():

        names = []
        # Search for matched faces and their corresponding encondings
        for encoding in encodings:
            stored_encodings.append(encoding)
            if len(stored_encodings) > 100:
                stored_encodings.pop(0)

        # Draw face label
        for ((x, y, w, h)) in faces:
            drawl_rect_label(frame, x, y, w, h, f"hooman, i have {len(stored_encodings)} samples", (0,255,0))

    cv2.imshow("Faces found",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import pandas as pd
pd.DataFrame(stored_encodings).to_csv("face_encodings.csv", header=None, index=None) # type: ignore