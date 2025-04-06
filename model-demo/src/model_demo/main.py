### Required libraries
import cv2
import numpy as np
import time
import PoseModule as pm
from PIL import Image

img = Image.open('C:\\Users\\Aaditya Khanal\\OneDrive\\Desktop\\posture-recognition\\model-demo\\src\\model_demo\\tbenchpress.png')

detector = pm.PoseDetector()

while True:
    img = cv2.imread(r"C:/Users/Aaditya Khanal/OneDrive/Desktop/posture-recognition/model-demo/src/model_demo/tbenchpress.png")
    img = detector.find_pose(img)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


