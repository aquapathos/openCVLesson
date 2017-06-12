# -*- coding: utf-8 -*-
import cv2
import numpy as np

imgfile = "scene.jpg" # 顔検出の対象画像名
frame = "polaroid.jpg" # ポラロイドのフレーム

# 入力画像とポラロイドフレーム画像の読み込み
src = cv2.imread(imgfile,1)
polaroid = cv2.imread(frame,1)
out = np.array((350,410,3))

# 回転行列の準備　回転角５度、スケーリング等倍
mat = cv2.getRotationMatrix2D((160,193),5,1.0)
mat[0,2],mat[1,2]=15,30  # 平行移動量をセット

# 幅 200, 高さ300 の部分をポラロイド枠の内部にコピー
(px1,px2,py1,py2) = (20,290,23,300)
(x1,y1,x2,y2) = (100,50,300,350)

ROI_POL = polaroid[py1:py2,px1:px2]
ROI_SRC = src[y1:y2,x1:x2]
cv2.resize(ROI_SRC,(px2-px1,py2-py1),ROI_POL)

# srcのROI を戻し、切り出した領域を赤で描く
cv2.rectangle(src, (x1,y1),(x2,y2), (0,0,255),2)

# polaroid のROI を戻し、ポラロイド画像を回転 → out
out = cv2.warpAffine(polaroid, mat, (370,420))

cv2.imshow("input",src)
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
