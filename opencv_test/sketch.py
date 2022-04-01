""" 
    Sketch for tunnel from face recognition to camera script 

    Info about Image subscriber: http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html 
    Query on help:
        * np.frombuffer(..) <- turns ROS image data into a proper cv2 image to be displayed
        * np.reshape(..)    <- turns the buffer array into a matrix
        * cv2.imshow(..)
        * cv2.cvtColor(..)
    
"""


import rospy 
from sensor_msgs.msg import Image 

import cv2

import numpy as np

def callback(self, image_data):
    """ callback to constanly refresh image through rospy.spin() ??? """

    # Convert imagen from sensor_msgs/Image format to cv2 format
    cv_image = np.frombuffer(
      image_data.data, 
      dtype=np.uint8
    ).reshape(image_data.height, image_data.width, -1)

    # ! ---

    database = DataBase()

    faces, names = [],[]

    FRAME_RATE = 30
    prev = time.time()
    while True:
        
        frame = cv_image  # ! OJO

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

        if (time.time() - prev) > 1./FRAME_RATE: 
            faces = face_recognition.face_locations(rgb)
            # the facial embeddings for face in inputP
            encodings = face_recognition.face_encodings(rgb, faces, model="large")
            
            # show faces
            if faces != ():

                names = []
                # Search for matched faces and their corresponding encondings
                for (encoding, (x,y,w,h)) in zip(encodings,faces):
                    names.append( database.recognize( Face(encoding, FaceRect(x,y,w,h,time.time()) ) ) )

            # Draw face label
            prev = time.time()

        for ((top, right, bottom, left), (name,dist)) in zip(faces,names):
            drawl_rect_label(gray, left, bottom, right - left, top - bottom, str(name) , (255,255,255))
                    
        cv2.imshow("Faces found",gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ! -- Archivos con modelo

cascPathfile = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathfile)

# ! -- Subscriptor

image_sub = rospy.Subscriber("/pepper/camera/ir/image_raw", Image, self.callback)

rospy.init_node('pepper_camera', anonymous=True)

# ! The node keeps listening to the camera topic until the process is canceled

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
    
cv2.destroyAllWindows()