#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:04:40 2021

@author: pyarena
"""

import cv2

import mediapipe as mp

import time



class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDect = mp.solutions.face_detection
        self.mpDrawing = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDect.FaceDetection(minDetectionCon)
        
    def findFaces(self, img):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = [] 
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                
                #mpDrawing.draw_detection(img, detection)
                #print(id, detection)
                #print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape 
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                                
                bboxs.append([id, bbox, detection.score])
               
                img = self.fDraw(img, bbox)
                
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 255),2)
        
        return img, bboxs
    
            
    def fDraw(self, img, bbox, l=30, t=10, rt=2):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
         
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #top left x,y
        cv2.line(img, (x,y), (x+l,y),(255, 0, 255), t)
        cv2.line(img, (x,y), (x,y+l),(255, 0, 255), t)
         #top right x1, y
        cv2.line(img, (x1,y), (x1 - l, y),(255, 0, 255), t)
        cv2.line(img, (x1,y), (x1,y + l),(255, 0, 255), t)
        #Bottom left x,y1
        cv2.line(img, (x,y1), (x+l,y1),(255, 0, 255), t)
        cv2.line(img, (x,y1), (x,y1-l),(255, 0, 255), t)
         #Bottom right x1, y1
        cv2.line(img, (x1,y1), (x1 - l, y1),(255, 0, 255), t)
        cv2.line(img, (x1,y1), (x1,y1 - l),(255, 0, 255), t)
          
        return img
    


def main():
    cap = cv2.VideoCapture('/home/pyarena/python/OpenCV/faceDetect/faceD1.mp4')
    pTime = 0
    detector = FaceDetector()
     
    while True:
        
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3  ,(0, 255, 0), 2) 
        cv2.imshow('Face Detection ', img)
        cv2.waitKey(10)


    

if __name__ == "__main__":
    main()
    
