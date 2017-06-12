# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name: surf.py SURF特徴抽出
# Purpose:
#
# Author: dan
#
# Created: 15/12/2016
# Copyright: (c) dan 2011,2012,2013,201,2015,2016
# Licence: <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import cv2

methods = ['FAST','MSER','AKAZE','KAZE','BRISK','ORB','SBD','SIFT','SURF']

def main():
  import sys

  try:
    imagename = sys.argv[1:2][0]
  except:
    imagename = "prof.jpg"
  print(imagename)

  colorImage = cv2.imread(imagename,1) # 1:color, 0:gray, -1:asis
  if colorImage == None:
    print("No Image")
    return
  grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

  FASTdtr = cv2.FastFeatureDetector_create()
  MSERdtr = cv2.MSER_create()
  AKAZEdtr = cv2.AKAZE_create()
  KAZEdtr = cv2.KAZE_create()
  BRISKdtr = cv2.BRISK_create()
  ORBdtr = cv2.ORB_create()
  SBDdtr = cv2.SimpleBlobDetector_create()
  SIFTdtr = cv2.xfeatures2d.SIFT_create()
  SURFdtr = cv2.xfeatures2d.SURF_create()

  resultImg = []
  met = 0
  for detector in (FASTdtr,MSERdtr,AKAZEdtr,KAZEdtr,BRISKdtr,ORBdtr,SBDdtr,SIFTdtr,SURFdtr):
    resultImg.append(colorImage.copy())
    keypoints = detector.detect(grayImage)
    #draw keypoints
    print("Image Keypoints: " ,len(keypoints))
    for ip in keypoints:
        pt = (np.round(ip.pt[0]).astype(int),np.round(ip.pt[1]).astype(int))
        radius = np.round(ip.size*0.25).astype(int)
        # print pt, radius
        cv2.circle(resultImg[met], pt,radius, (255,0,0),1,8,0)

    cv2.imshow(methods[met], resultImg[met])
    met += 1
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
