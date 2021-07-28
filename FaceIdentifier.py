import cv2, os
import numpy as np
from PIL import Image
import pickle
import sqlite3

cascadePath = 'haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read("Face_Recognizer/Face_Training_Data.yml")

#path = "Known_Faces"

def getProfile(id):
    conn = sqlite3.connect("FaceDataBase.db")
    cmd = "SELECT * from PEople WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,225,0),2)
        profile = getProfile(id)
        if(profile != None):
            cv2.putText(img,str(profile[1]),(x,y+h+30), font, 0.8,(0,0,255),2)
            cv2.putText(img,str(profile[2]),(x,y+h+60), font, 0.8,(0,0,255),2)
            cv2.putText(img,str(profile[3]),(x,y+h+90), font, 0.8,(0,0,255),2)
            #cv2.putText(img,str(profile[3]),(x,y+h+120), font, 0.8,(0,0,255),2)

    cv2.imshow('video',img)
    cv2.waitKey(10)
