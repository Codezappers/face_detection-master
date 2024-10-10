import face_recognition
import cv2
import numpy as np

#Get a reference to the default webcam
video_capture = cv2.VideoCapture(0)

#Load a sample picture and learn how to recognize it
your_image = face_recognition.load_image_file("ryan.png")
your_face_encoding = face_recognition.face_encodings(your_image)[0]

#Create an array of known face encodings
known_face_encodings = [
    your_face_encoding,
]

known_face_names = [
    "Ryan"
]

#Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #grab a single frame of the video
    ret, frame = video_capture.read()
    
    #Only process every other frame
    if process_this_frame:
        #Resize frame of the video to 1/4 size for faster 
        #face recognition processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, y=0.25)
        
        #Convert the image from BGR (which OpenCV uses) to RGB (face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        #Find all the faces and facial encodings in the current frame of the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
    
