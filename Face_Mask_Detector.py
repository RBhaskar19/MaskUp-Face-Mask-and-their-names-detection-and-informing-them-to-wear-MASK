# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import pyttsx3



def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


# ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Face_Recognizer/Face_Training_Data.yml')

face_cascade_Path = "haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(face_cascade_Path)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
Ids = []
# names related to ids: The names associated to the ids: 1 for Mohamed, 2 for Jack, etc...
names = ['None', 'Bhaskar','Susanna']  # add a name into this list
# --------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')

engine.setProperty('voice',voices[0].id)


#-----------------
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
MaskNet = load_model("mask_detector.model")

# initialize the video stream
print("Starting video stream...")
cam = cv2.VideoCapture(0)  # Starts VideoStream
vs = VideoStream(0)
cam.set(3, 250)
cam.set(4, 250)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    start_point = (15, 15)
    end_point = (370, 80)
    thickness = -1
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=400)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, MaskNet)

 # -------------------------------------------------------------------------------
    #resultImg, faceBoxes = highlightFace(faceNet1, frame)
 # -------------------------------------------------------------------------------
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text

        label = "Mask" if mask > withoutmask else "No Mask"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)

        # display the label and bounding box rectangle on the output

        # frame

        cv2.putText(frame, label, (startX, startY - 10),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
 # -----------------
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if(label == 'No Mask'):
                if (confidence < 100):
                    if names[id] not in Ids:
                        Ids.append(names[id])
                
                else:
                # Unknown Face
                    id = 0
                

            cv2.putText(img, str(names[id]), (x + 5, y - 5),
                        font, 1, (255, 255, 255), 2)

        cv2.imshow('Camera', img)
 # -------------

        # include the probability in the label
        if(label == 'No Mask'):
            if len(Ids)<=3:
                for i in Ids:
                    audio = "Hello"+str(i)+"please wear mask"
                    engine.say(audio)
                    engine.runAndWait()

        elif(label == 'Mask'):
            try:
                if(len(Ids)>0):
                    Ids.remove(names[id])
                
            
            except ValueError as error:
                pass

            	

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
print(Ids)
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
