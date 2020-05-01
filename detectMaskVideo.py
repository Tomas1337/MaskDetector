#construct the argument parse and parse the arguments
from imutils.video import VideoStream
import os
import torch
from fastai.vision import *
import imutils
import time
import cv2
from LoadPredict import *
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
#prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
prototxtPath =  "D:/Projects/Mask Detector/face_detector/deploy.prototxt"
#weightsPath = os.path.sep.join([args["face"],
	#"res10_300x300_ssd_iter_140000.caffemodel"])

weightsPath = "D:/Projects/Mask Detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
#faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our traced mask detector model from disk
print("[INFO] loading mask detector model...")

maskNet = maskPredictor_fastai()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_count = 0
# loop over the frames from the video stream
while True:
	results = []
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	

	#Conduct detections
	face_box = detect_face(frame, faceNet)
	
	

	#Check if there are any detected faces
	#if face_box.shape[0] < 10 or face_box.shape[1] < 10:
	#	continue

	#Predict if Face has Mask or Not
	results = detect_mask(face_box, frame, maskNet)


	# show the output frame
	cv2.namedWindow("Frame")
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
'''
#For Showing Face
	cv2.namedWindow("Face")
	cv2.imshow("Face", face)
	key = cv2.waitKey(1) & 0xFF
'''
	# if the `q` key was pressed, break from the loop
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stream.release()
vs.stop()