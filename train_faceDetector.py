import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "Known_Faces"

def getImagesWithIds(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagepath in imagepaths:
        faceImg = Image.open(imagepath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        Id = int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(Id)
        cv2.imshow("trainning",faceNp)
        cv2.waitKey(10)
    return Ids, faces

Ids, faces = getImagesWithIds(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('Face_Recognizer/Face_Training_Data.yml')
cv2.destroyAllWindows()
