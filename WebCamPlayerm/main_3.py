# Class Files
from imutils.video import VideoStream
import torch
import numpy as np
import time
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from fastai.vision import *
import youtube_dl
import imutils
import pathlib
import scipy.misc
import uuid
from multiprocessing import Process, Manager, Queue
import multiprocessing as mp
import dlib

#Class for a Mask Predictor
class maskPredictor_fastai:
    def __init__(self):
        #defaults.device = torch.device('cpu')
        classes = ['noMask','withMask']
        path = 'C:/Projects/Mask Detector/train_data'
        
        #Create blank databunch
        data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=128).normalize(imagenet_stats)
        
        #Create the learner
        learn = cnn_learner(data2, models.resnet34, metrics=error_rate)
        #Load it with the pretrained RESNET Weights 
        #'res34_stage2_1 best one ye
        #'res50_stage1_v3 best one yet
        learn.load('res34_stage1_v1')
        #learn.load('res50_stage1_v5')
        #learn.load('vgg19_stage1_v1')
        self.model = learn
        
    def maskPredict(self,face):
        img = Image(pil2tensor(face, dtype=np.float32).div_(255))
        tic = time.perf_counter() 
        #Make prediction
        while True:
            try:
                pred_class,pred_idx,outputs = self.model.predict(img)
                score = outputs.max()
                break
                
            except ZeroDivisionError:
                print('Computation Error')
                score = 0
                pred_idx = 0
                return None

        toc = time.perf_counter()       
        #print(f"Mask Inference:{toc - tic:0.4f}s")
        return score, pred_idx
        
#class for Face Detection using FastMTCNN
class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
    
    def __call__(self, frames):
        #print("I, FastMTCNN, have been called")
        if self.resize != 1:
            frames = cv2.resize(frames, (int(frames.shape[1] * self.resize), int(frames.shape[0] * self.resize)))
        
        #results = self.mtcnn.detect(frames[::self.stride])
        tic= time.time() 
        boxes, results = self.mtcnn.detect(frames, landmarks=False)
        toc=time.time()
    
        #print('Face Detection Inference is:' + str(toc-tic))
        return [boxes, results]


#class for Face Detection using Standard MTCNN    q    
class stdMTCNN(object):  
    def __init__(self, resize=1, *args, **kwargs):
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
    
    def __call__(self, frames):
        if self.resize != 1:
            frames = cv2.resize(frames, (int(frames.shape[1] * self.resize), int(frames.shape[0] * self.resize)))
        
        #results = self.mtcnn.detect(frames[::self.stride])
        tic= time.time() 
        boxes, results = self.mtcnn.detect(frames, landmarks = False)
        toc=time.time()
        print('Face Detection Inference is:' + str(toc-tic))
        
        return [boxes, results]

#Load in prediction Functions

### Face
def detect_face(frame, faceNet, draw = True):
    # initialize our results list
    results = []
    
    #The higher the better face detections works
    img_size = 300
    
    if (frame is None):
        print('No frame detected')
        return None
    # grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    
    # pass the blob through the network and obtain the face detections
    blob = cv2.dnn.blobFromImage(frame, 1.0, (img_size, img_size),(104.0, 177.0, 123.0))
    # Pass frame through Input
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2] 
        minConf = 0.6 
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        #compute the (x, y)-coordinates of the bounding box for

        if confidence > minConf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #Detect if any negative values and turn to 0
            box[box < 0] = 1
            
            (startX, startY, endX, endY) = box.astype("int")
            #Store results of face coordinates
            d = [startX, startY, endX, endY]
            
            
        
            #Draw rectangle around face
            if draw is True:
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #In cases that there are more than one face per frame, it appends to a list
            d = [startX, startY, endX, endY]
            
            if any(t < 0 for t in d):
                for i,p in enumerate(d):
                    print (i)
                    if d[i] < 0:
                        d[i] = 0
            results.append(d)       
    return results

def detect_face_MTCNN(frame, faceNet, draw = True):
    results = []
  
    if (frame is None):
        print('No frame detected')
        return None
    
    (h, w) = frame.shape[:2]
    pad_f = 0.05
    
    frame_p = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_p = frame
    minConf = 0.5
    

    #Check if any objects are tracked; If none, conduct a detect Face Algorithm

        #Pass frame through network and come out with bounding box in (face)
        #We do the predictions here
    [face, score] = faceNet(frame_p)
    if (face is None or face.size == 0):
        print('No face detected')
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        return None

    #Change negative values to 0 (Bug in in the model)
    face[face < 0] = 0

    #Resize facebounding box to original size
    if faceNet.resize != 1: 
        face = np.multiply(face,(1/faceNet.resize))
    
    #For each detected face
    for i in range(0, face.shape[0]):

        #Check if results is more thatn confidence
        if score[i] > minConf:
            # Unpack
            (startX, startY, endX, endY) = face[i].astype(int)
            pad_x = int((endX-startX) * pad_f)
            pad_y = int((endY-startY) * pad_f)

            #Draw rectangle around face
            if draw:
                cv2.rectangle(frame, (startX-pad_x, startY-pad_y), (endX+pad_x, endY+pad_y),(0, 0, 255), 2)

            d = [startX-pad_x, startY-pad_y, endX+pad_x, endY+pad_y]
            #d = [startX, startY, endX, endY]
            results.append(d)
        
    return results
    

### MASK
def detect_mask(face_box, frame, maskNet, save = False, minConf = 0.8, draw = True):
    pred_dict = {0: 'No Mask',1: 'With Mask'}
    results = []
    p = 0
    frame_p= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   #Check if face has data
    if [x for x in (face_box, frame_p) if x is None]:
        print('No face detected 1')
        return None
    
    #Iterate over the returned results
    for i in face_box:
        (startX, startY, endX, endY) = face_box[p]
        p = p+1
        #Crop out just the face from the frame
        face = frame_p[startY:endY, startX:endX]
        if [x for x in (face, frame) if x is None]:
            print('No face detected 2')
            return None
        
        #Pass through Mask Detector Model and Time
        mask_results = maskNet.maskPredict(face)
        
        if mask_results is None:
            return None

        #Get results from tensors and convert to a list
        d = [mask_results[1].item(), mask_results[0].item()]

        #Only annotate ones which have a good confidence
        if d[1] > minConf:
        
        #Script for saving image in folder
            if save:
                print('called')
                size = (224, 224)
                face = cv2.resize(face, size)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                im = Image.fromarray(face)

                if d[0] == 1:#If Face has Mask
                    outfile = '%s/%s.jpg' % ('train_data/withMask', 'withMask' + str(uuid.uuid4()))
                    im.thumbnail(size)
                    im.save(outfile)
                    print('saved image')
                elif d[0] == 0: #If face has noMask
                    outfile = '%s/%s.jpg' % ('train_data/noMask', 'noMask' + str(uuid.uuid4()))
                    im.thumbnail(size)
                    im.save(outfile)
                    print('saved image')

            #Annotate if draw is True
            if draw:

                #Change color of text according to results
                print('annotate')
                if d[0] == 1:
                    text_color = (0,255,0)
                elif d[0] == 0:
                    text_color = (0,0,255)

                text = "{}: {:.2f}%".format(pred_dict[d[0]], d[1] * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 2)

        results.append(d)

    return results


class WebCamPlayer:    
    def __init__(self, faceNet='ResCaffe'):
        print("Initializing")
        #Load Predict setup files
        current_dir = pathlib.Path().absolute()
        
        #Choosing the face Detection System
        if faceNet == 'ResCaffe':
            #Load up hass Face Detector    
            prototxtPath =  os.path.join(current_dir,'face_detector','deploy.prototxt')
            weightsPath = os.path.join(current_dir,'face_detector','res10_300x300_ssd_iter_140000.caffemodel')
            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            self.faceNetType = 'ResCaffe' 
        
        elif faceNet == 'SFD':
            #Load up hass SFD Detector   
            prototxtPath =  os.path.join(current_dir,'face_detector','deploy_sfd.prototxt')
            weightsPath = os.path.join(current_dir,'face_detector', 'SFD.caffemodel')
            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            self.faceNetType = 'SFD' 
        
        elif faceNet == 'fastMTCNN':
            faceNet = FastMTCNN(
                stride=4,
                resize=0.5,
                margin=14,
                factor=0.6,
                keep_all=True,
                select_largest=False,
                device='cpu'
            )
            self.faceNetType = 'fastMTCNN'
            self.resize = faceNet.resize
            
        elif faceNet == 'stdMTCNN':
            faceNet = stdMTCNN(
                select_largest=False,
                resize=1,
                margin=1,
                factor=0.6,
                keep_all=True,
                device='cpu'
            )
            self.faceNetType = 'stdMTCNN'
            self.resize = faceNet.resize
            
        print(str(self.faceNetType)+' initialized as the Detection')
        #Choosing the Mask Detection Version
        maskNet = maskPredictor_fastai()
        
        self.faceNet = faceNet
        self.maskNet = maskNet
        
    
    def webcam_play(self, detect = True, draw = True, save = False, skip_frames = 5):
        vs = VideoStream(src=1).start()
        #vs = VideoStream(src =1)
        #src = 1
        #stream = cv2.VideoCapture(src)
		
        print("Starting Web Cam Feed")
        time.sleep(2.0)
        frame_count = 0
    
        M = 0
        while True:
            frame_count = frame_count + 1
            
            # grab the frame from the video stream
            # to have a maximum width of 400 pixels
            frame = vs.read()
            
            if (frame is None):
                print('No frame detected')
                break
            #Resize The frame
            frame = imutils.resize(frame, width=800)
            if detect:
                if self.faceNetType in ['ResCaffe','SFD']:
                    face_box = detect_face(frame, self.faceNet)
                    if face_box is None:
                        continue
                    results = detect_mask(face_box, frame, self.maskNet,save)

                elif self.faceNetType in ['fastMTCNN','stdMTCNN']:
                    face_box = detect_face_MTCNN(frame, self.faceNet, draw)
                    if face_box is None:
                        continue
                    results = detect_mask(face_box, frame, self.maskNet)

            # display frame
            cv2.namedWindow("frame")
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): 
                vs.stop()
                vs.stream.release()
                break
                    
        # release VideoCapture
        vs.stop()
        vs.stream.release()



vid = WebCamPlayer(faceNet = 'stdMTCNN')    
vid.webcam_play(draw=True)
