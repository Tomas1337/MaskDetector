# MaskDetector
Real Time Mask Detector

A simple pipeline to detect whether a person is wearing a mask or not in a Video Stream.
The Pipeline first detects faces then passes those faces on to a secondary model which predicts whether that face is wearing a mask or not.


Only works right now on the Jupyter Notebook.
By Default, it uses an External Camera (src=1)
Change src=0 to use built in camera on 'Play webcam stream'
vs = VideoStream(src=1).start()


Update 5/1/2020
Added a module to stream youtube video and live apply the detection algorithsm
Added capability to switch between Face Detection models on call. ResCaffe has a good FPS but medium performance on face detection. MTCNN is great at face detection but is slow. FastMTCC gives good face detection and is a bit faster. Recommend to use MTCNN when collecting data or with use of GPUs. Recommend to use ResCaffe for standard application

Although, there is only one 

Download prediction model and place on root/data/models
https://drive.google.com/open?id=1ZXxYqsO-MlkjIgeS8caVvO1J8QENF4Eb
To be done:


1 Increase training data size (Trained on fast ai resnet34)

2. Deploy on web app

3. Make asyncrohous


