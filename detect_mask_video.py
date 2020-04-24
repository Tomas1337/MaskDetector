# USAGE
# python detect_age_video.py --face face_detector --age age_detector

# import the necessary packages
from imutils.video import VideoStream
from LoadPredict import *
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, mask_model, minConf=0.5):
	global frame_count
	# define the list of age buckets our age detector will predict
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# initialize our results list
	results = []

	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > minConf:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the ROI of the face
			face = frame[startY:endY, startX:endX]

			##Try a trained model with normalized faces

			# ensure the face ROI is sufficiently large
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			# construct a blob from *just* the face ROI

			#toc1 = time.perf_counter()
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)
            
            #Process Prediction
			#Find a way for this to be an asynchronous task
			mask_results = maskNet.maskPredict(frame)
			mask_prediction = mask_results[1].item()
			mask_score = mask_results[0].item()
           
			# construct a dictionary consisting of both the face
			# bounding box location along with the age prediction,
			# then update our results list
			d = {
				"loc": (startX, startY, endX, endY),
                "mask": (mask_prediction, mask_score)
			}
			results.append(d)

	# return our results to the calling function
	return results

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our traced mask detector model from disk
print("[INFO] loading mask detector model...")
#mask_model = loadtracedModel('squeezeNet.pt')
maskNet = maskPredictor() 
maskNet.loadtrace('modelResNet34.pt')


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
pred_dict = {0: 'No Mask',1: 'With Mask'}
frame_count = 0
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	# detect faces in the frame, and for each face in the frame,
	# predict the age
	results = detect_and_predict_age(frame, faceNet, maskNet,
		minConf=args["confidence"])

	# loop over the results
	for r in results:
		# draw the bounding box of the face along with the associated
		# predicted age
		
		mask_predict = r["mask"][0]
		text = "{}: {:.2f}%".format(pred_dict[mask_predict], r["mask"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
