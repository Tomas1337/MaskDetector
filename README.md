# MaskDetector
Real Time Mask Detector

To run use command: python detect_mask_video.py --face face_detector 

By Default, it uses an External Camera
Change src=0 to use built in camera on line 105 of detect_mask_video.py

vs = VideoStream(src=1).start()

Download prediction model and place on root
https://drive.google.com/file/d/1lR5EgA93mW9kzcAW1MTJybat9PfRe33V/view?usp=sharing
