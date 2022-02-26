import cv2
import pickle
import face_recognition
import os
import time
from typing import Any, Dict, List, Tuple

def drawl_rect_label(cv2_image : Any, x : int, y : int, w : int, h : int, label : str, color : Tuple[int, int, int] = (0,255,0)):
    """
        Try to draw a rectangle with a label in an image
    """
    image = cv2.rectangle(cv2_image, (x,y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

class DataBase:

    def __init__(self) -> None:
        self._stored_embeddings : Dict[int, Any] = {}
        self._id_counter = 0
        pass

    def add(self, embeddings : Any, id : int):
        self._stored_embeddings[id] = embeddings

    def get(self, embeddings : Any) -> str:
        for (k, v) in self._stored_embeddings.items():
            if True in face_recognition.compare_faces([v], embeddings):
                return k

        return None

    def recognize(self, embeddings : Any) -> int:
        """
            Check if in database. If not, add it with a new id, 
            otherwise, return current id and update embeddings
        """

        if (elem := self.get(embeddings)):
            self._stored_embeddings[elem] = embeddings
            return elem
        
        new_id = self._id_counter

        self.add(embeddings, new_id)
        self._id_counter += 1

        return new_id

        

cascPathfile = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathfile)

print("Starting capture")
video_capture = cv2.VideoCapture(0)
database = DataBase()
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
    encodings = face_recognition.face_encodings(rgb)

    # show faces
    if faces != ():

        names = []
        # Search for matched faces and their corresponding encondings
        for encoding in encodings:
            names.append(database.recognize(encoding))

        # Draw face label
        for ((x, y, w, h), name) in zip(faces,names):
            drawl_rect_label(frame, x, y, w, h, str(name), (0,255,0))

    cv2.imshow("Faces found",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


        