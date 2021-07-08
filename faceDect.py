#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:23:30 2021

@author: pyarena
"""

import cv2

import mediapipe as mp

import time


cap = cv2.VideoCapture('/home/pyarena/python/OpenCV/faceDetect/faceD1.mp4')
pTime = 0

mpFaceDect = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDect.FaceDetection(0.75)

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDrawing.draw_detection(img, detection)
            #print(id, detection)
            #print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape 
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                        (bbox[0], bbox[1] - 20),cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 255),2)
            
            
    cTime = time.time() # frame rate
    fps_rate = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3  ,(0, 255, 0), 2) 
    cv2.imshow('Face Detection ', img)
    cv2.waitKey(10)
    













