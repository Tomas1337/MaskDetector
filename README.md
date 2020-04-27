# MaskDetector
Real Time Mask Detector

To run use command: python detect_mask_video.py --face face_detector 

By Default, it uses an External Camera

Change src=0 to use built in camera on line 105 of detect_mask_video.py

vs = VideoStream(src=1).start()

Download prediction model and place on root

https://drive.google.com/file/d/1NKXeHM8NKSIb0wSnG6vij30afkY6arZK/view?usp=sharing

To be done:


1 Increase training data size (Trained on fast ai resnet34)

2. Deploy on web app

3. Make asyncrohous


