# -*- coding: utf-8 -*-
import cv2
import numpy as np

imgfile = "scene.jpg" # 顔検出の対象画像名
frame = "polaroid.jpg" # ポラロイドのフレーム

# 入力画像とポラロイドフレーム画像の読み込み
src = cv2.imread(imgfile,1)
orig = 1*src
polaroid = cv2.imread(frame,1)
out = np.array((350,410,3))

# 回転行列の準備　回転角５度、スケーリング等倍
mat = cv2.getRotationMatrix2D((160,193),5,1.0)
mat[0,2],mat[1,2]=15,30  # 平行移動量をセット

# 幅 200, 高さ300 の部分をポラロイド枠の内部にコピー
(px1,px2,py1,py2) = (20,290,23,300)
(x1,y1,x2,y2) = (100,50,300,350)

ROI_POL = polaroid[py1:py2,px1:px2]

def on_change(xx1,yy1,xx2,yy2):
    global x1,y1,x2,y2,src
    x1,y1,x2,y2 = xx1,yy1,xx2,yy2
    if x1 > x2:
        xx2, xx1 = x1, x2
    if y1 > y2:
        yy2 ,yy1 = y1, y2
    src = 1*orig
    ROI_SRC = src[yy1:yy2,xx1:xx2]
    if not(xx1==xx2 or yy1==yy2):
        cv2.resize(ROI_SRC,(px2-px1,py2-py1),ROI_POL)
    # srcを初期化し切り出した領域を赤で描く
    cv2.rectangle(src, (xx1,yy1),(xx2,yy2), (0,0,255),2)
    # polaroid のROI を戻し、ポラロイド画像を回転 → out
    out = cv2.warpAffine(polaroid, mat, (370,420))
    cv2.imshow("input",src)
    cv2.imshow("result",out)

def on_changeX1(x):
    global x1
    x1 = x
    on_change(x1,y1,x2,y2)

def on_changeY1(y):
    global y1
    y1 = y
    on_change(x1,y1,x2,y2)


on_change(x1,y1,x2,y2)
cv2.createTrackbar("ThresX1","input",x1,src.shape[1]-1,on_changeX1)
cv2.createTrackbar("ThresY1","input",y1,src.shape[0]-1,on_changeY1)

cv2.waitKey(0)
cv2.destroyAllWindows()
