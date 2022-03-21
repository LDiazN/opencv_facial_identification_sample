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

# Anadir criterio de inhabilitacion para la edad
class FaceRect:
    """
        Represents a face rectangle with an update time
    """

    def __init__(self,x : int, y : int, width : int, height : int, age : float) :
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._age = age

    @property
    def age(self) -> float:
        return self._age

    def as_diagonal(self) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        return ((self._x,self._y, self._x + self._width), (self._y + self._height))

    def vertices(self) -> List[Tuple[int,int]]:
        ll,ur = as_diagonal(self)
        ul, lr = (ll[0], ll[1] + self._height) , (ur[0], ur[1] - self.height)
        return [ll,ur,ul,lr]
    
    def overlap(self, rect_2 : Self) -> bool:
        ll, ur = as_diagonal(self) 

        for ref in vertices(rect_2):
            if within(ll,ur,ref) :
                return true
        
        return false

    def within (ll : Tuple[int,int], ur : Tuple[int,int], ref : Tuple[int,int]) -> bool:
        return (ll[0] <= ref[0] and ll[1] <= ref[1]) or (ref[0] <= ur[0] and ref[1] <= ur[1])

class Face:
    """
        Represents a face object and provides a way to compare faces
    """
    def __init__(self, face_embeddings : numpy.ndarray,  rectangle : FaceRect, tolerance: float = 0.6):
        self._face_embeddings = face_embeddings
        self._tolerance = tolerance
        self._rectangle = rectangle 
    
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

    # extender comparacion con rectangulos y sus criterios
    def compare_with_rect_info(self, face_2 : Self) -> Tuple[bool, float]:
        """ 
            Use to distinguish between faces by distance and embedding criteria:
            First check if face embeddings are whithin and acceptable distance (they match).
            If so, check if they're the same face by checking overlaping over their face rectangles.
        """
        distance = face_recognition.face_distance([self.face_embeddings], face_2.face_embeddings)[0]



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
        for (encoding,(x,y,w,h)) in zip(encodings,faces):
            names.append(database.recognize( Face(encoding,FaceRect(x,y,w,h,time.time()))))

        # Draw face label
        for ((x, y, w, h), (name,dist)) in zip(faces,names):
            drawl_rect_label(frame, x, y, w, h, str(name)+" "+str(dist), (0,255,0))

    cv2.imshow("Faces found",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


        