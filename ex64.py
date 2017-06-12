# -*- coding: utf-8 -*-
import cv2
import numpy as np
#import matplotlib.pyplot as plt

imgfile = "scene.jpg" # 顔検出の対象画像名
frame = "polaroid.jpg" # ポラロイドのフレーム

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# 入力画像とポラロイドフレーム画像の読み込み
src = cv2.imread(imgfile)
orig = src.copy()
polaroid = cv2.imread(frame)
#out = np.zeros((340,410),dtypes=np.uint8)

# 回転行列の準備　回転角５度、スケーリング等倍
mat = cv2.getRotationMatrix2D((160,193),5,1.0)
mat[0,2],mat[1,2]=15,30
# 平行移動量をセット

def on_change(x1,y1,x2,y2):
  # 入力画像 org から、座標(x,y) を起点として
  # 幅 w, 高さh の部分をポラロイド枠の内部にコピー
  polaroid[23:300,20:290]= cv2.resize(orig[y1:y2,x1:x2],(270,277))
  src = orig.copy()
  cv2.rectangle(src, (x1,y1),(x2,y2), (255,0,0),2)
  # polaroid のROI を戻し、ポラロイド画像を回転 → out
  out = cv2.warpAffine(polaroid, mat, (370,420))
  cv2.imshow("input",src)
  cv2.imshow("result",out)
  # plt.subplot(1,2,1)
  # plt.imshow(cv2.cvtColor(src,cv2.COLOR_RGB2BGR))
  # plt.subplot(1,2,2)
  # plt.imshow(cv2.cvtColor(out,cv2.COLOR_RGB2BGR))


# 学習済みの顔検出器を用いて、顔を検出
cascade_file = "data/haarcascades/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
faces = detect(orig, cascade)

for (x1,y1,x2,y2) in faces:
  on_change(x1,y1,x2,y2)
  cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
