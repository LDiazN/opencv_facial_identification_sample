from typing_extensions import Self
import cv2
import pickle
import face_recognition
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy

def drawl_rect_label(cv2_image : Any, x : int, y : int, w : int, h : int, label : str, color : Tuple[int, int, int] = (0,255,0)):
    """
        Try to draw a rectangle with a label in an image
    """
    image = cv2.rectangle(cv2_image, (x,y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

class Face:
    """
        Represents a face object and provides a way to compare faces
    """
    def __init__(self, face_embeddings : numpy.ndarray, tolerance: float = 0.6):
        self._face_embeddings = face_embeddings
        self._tolerance = tolerance
    
    @property
    def face_embeddings(self) -> numpy.ndarray:
        """
            Return face's embeddings
        """
        return self._face_embeddings

    @property
    def tolerance(self) -> float:
        """
            Return tolerance value
        """
        return self._tolerance

    def compare(self, face_2 : Self) -> bool:
        """
            Checks if two faces are the same
        """
        return self.compare_and_distance(face_2)[0]

    def distance(self, face_2 : Self) -> float:
        """
            Check distance from this instance to other instance
        """
        return self.compare_and_distance(face_2)[1]

    def compare_and_distance(self, face_2 : Self) -> Tuple[bool, float]:
        """
            Check if this instance is the same as other and return it's distance
        """
        distance = face_recognition.face_distance([self.face_embeddings], face_2.face_embeddings)[0]

        return (distance < self.tolerance, distance)

class DataBase:

    def __init__(self) -> None:
        self._stored_embeddings : Dict[int, Face] = {}
        self._id_counter = 0

    def add(self, face : Face, id : int):
        self._stored_embeddings[id] = face

    def add_new(self, face : Face):
        id = self._id_counter
        self._id_counter += 1
        self.add(face, id)

    def get(self, face : Face) -> Optional[Tuple[int,float]]:
        best : float = 10000000
        out = -1
        
        for (k,v) in self._stored_embeddings.items():
            match, dist = v.compare_and_distance(face)
            if match and dist < best:
                best = dist
                out = k

        if out >= 0:
            return (out,best)
        return None 
        
    def recognize(self, face : Face) -> Tuple[int, float]:
        """
            Check if in database. If not, add it with a new id, 
            otherwise, return current id and update embeddings
        """

        if (elem := self.get(face)) != None:
            self._stored_embeddings[elem[0]] = face
            return elem      

        new_id = self._id_counter

        self.add(face, new_id)
        self._id_counter += 1

        return (new_id,0)

        

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
    encodings = face_recognition.face_encodings(rgb, faces)

    # show faces
    if faces != ():

        names = []
        # Search for matched faces and their corresponding encondings
        for encoding in encodings:
            names.append(database.recognize(Face(encoding)))

        # Draw face label
        for ((x, y, w, h), (name,dist)) in zip(faces,names):
            drawl_rect_label(frame, x, y, w, h, str(name)+" "+str(dist), (0,255,0))

    cv2.imshow("Faces found",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


        