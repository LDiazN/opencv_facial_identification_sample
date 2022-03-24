"""
    Implements a face memory manager to store data in runtime about faces and perform
    common operations over them
"""

from typing_extensions import Self
import cv2
import pickle
import face_recognition
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy

LIKELYHOOD_TRESHOLD = 0.6

class Face:
    """
        Represents a face object and provides a way to compare faces
    """
    def __init__(self, face_embeddings : numpy.ndarray, tolerance: float = 0.6, stored_samples : int = 31, variance_multiplier : float = 1.2):
        self._face_embeddings = face_embeddings
        self._tolerance = tolerance
        self._last_samples  = []
        self._stored_samples = stored_samples
        self._variance_multiplier = variance_multiplier

    @property
    def face_embeddings(self) -> numpy.ndarray:
        """
            Return face's embeddings
        """
        return self._face_embeddings

    @property
    def mean_face_vector(self) -> numpy.ndarray:

        if len(self._last_samples) < self._stored_samples:
            return self._face_embeddings

        return (1/self._stored_samples) * sum(self._last_samples)

    @property
    def distance_variance_and_mean(self) -> Tuple[float, float]:

        n_samples = len(self._last_samples)
        distances = numpy.array([numpy.linalg.norm(self._last_samples[i] - self._last_samples[i+1]) for i in range(n_samples - 1)])
        distance_variance = numpy.var(distances)
        distance_mean = numpy.mean(distances)
        return distance_variance, distance_mean


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
        distance = face_recognition.face_distance([self._face_embeddings], face_2.face_embeddings)[0]

        return (distance < self.tolerance, distance)

    def update(self, face_2 : Self):
        """
            Update data about this face from another one
        """
        self._face_embeddings = face_2.face_embeddings
        self._last_samples.append(face_2.face_embeddings)
        if len(self._last_samples) > self._stored_samples:
            self._last_samples.pop(0)

class DataBase:

    def __init__(self) -> None:
        self._stored_embeddings : Dict[int, Face] = {}
        self._id_counter = 0

    def add(self, face : Face, id : int):
        self._stored_embeddings[id] = face

    def add_new(self, face : Face) -> int:
        """
            Add a new face asiggning it a new id and returning such id 
        """
        id = self._id_counter
        self._id_counter += 1
        self.add(face, id)
        return id 

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
        
    def replace(self, key : int, new_face : Face):
        """
            Change existent face with key 'key' with face 'face'
        """
        self._stored_embeddings[key].update(new_face)


    def recognize(self, face : Face) -> Tuple[int, float]:
        """
            Check if in database. If not, add it with a new id, 
            otherwise, return current id and update embeddings
        """

        if (elem := self.get(face)) != None:
            self.replace(elem[0], face)
            return elem      

        new_id = self._id_counter

        self.add(face, new_id)
        self._id_counter += 1

        return (new_id,0)

    def recognize_many(self, faces : List[Face]) -> List[Tuple[int, float]]:
        """
            For each face, return its id and distance to the predicted face
        """
        MAX_VAL = 10000000

        # If no known faces, just create a new one for each
        if not self._stored_embeddings:
            results = [(self.add_new(face), 0) for face in faces]
            return results
            
        # Create a matrix with distance between each known face and new face
        items = list(self._stored_embeddings.items())
        face_embeddings = [face.face_embeddings for (_,face) in items]

        # Matrix with distance between each known face to each new face
        distances_matrix = numpy.array([face_recognition.face_distance(face_embeddings, face.face_embeddings) for face in faces])
        # Matrix that tells if this faces do match 
        less_than_tolerance = numpy.vectorize(lambda x: x < LIKELYHOOD_TRESHOLD)
        matching_matrix = less_than_tolerance(distances_matrix)
        min_per_col = [numpy.argmin(c) for c in distances_matrix.T]

        change = True
        while change:
            change = False

            # Tie break, if two columns contain the same min, then set the highest of both as infinite
            for (i, min_row_1) in enumerate(min_per_col):
                for j  in range(i+1, len(min_per_col)):
                    min_row_2 = min_per_col[j]
                    if min_row_1 == min_row_2:
                        if distances_matrix[min_row_1][i] < distances_matrix[min_row_2][j]:
                            distances_matrix[min_row_2][j] = MAX_VAL
                        else:
                            distances_matrix[min_row_1][i] = MAX_VAL
                        change = True

            min_per_col = [numpy.argmin(c) for c in distances_matrix.T]
            matching_matrix = less_than_tolerance(distances_matrix)
                
        # Now that we have unique mins, we have to check which rows require a new id 
        results : List[Tuple[int, float]]= [(-1,-1.0) for _ in range(len(faces))]
        for (i, row) in enumerate(matching_matrix):
            if not any(row):
                new_id = self._id_counter
                self.add_new(Face(face_embeddings[i]))
                results[i] = (new_id, 0)

        # now assign nearest for those that are min
        for (col_index, row_index) in enumerate(min_per_col):
            if matching_matrix[row_index][col_index]:
                results[row_index] = (items[col_index][0], distances_matrix[row_index, col_index])
        
        assert all(r != (-1, -1) for r in results)
        return results
