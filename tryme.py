import cv2
import numpy as np

fname = "prof.jpg"   # ←　任意の画像ファイルに変更せよ
# カラー画像としてイメージを読み込み
src = cv2.imread(fname,cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR = 1

# 入力画像と同サイズの8ビット、１プレーン分の容器を用意
(height,width,_) = src.shape
img1 = np.zeros((height,width),np.uint8)
img2 = img1.copy()
img3 = img1.copy()
srcGray = img1.copy()

# 入力画像と同形式の出力画像用の容器を用意
out = src.copy()

# カラー画像をグレイスケール画像へ変換
srcGray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

# グレー画像をガウシアンフィルタで平滑化
srcGray = cv2.GaussianBlur(srcGray,(5,5),1)

thres = 90
offset = 10
# ２種類のしきい値処理で2値化
# 27行目〜35行目を次のように修正
def show(wname,img):
    cv2.namedWindow(wname,cv2.WINDOW_KEEPRATIO)
    cv2.imshow(wname,img)

def on_changeX(x):
    global thres
    thres = x
    gosei()

def on_changeY(y):
    global offset
    offset = y
    gosei()

def gosei():
    ret,img1 = cv2.threshold(srcGray,thres,255,cv2.THRESH_BINARY)
    img2 = cv2.adaptiveThreshold(srcGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, offset)
    img3 = cv2.bitwise_and(img1,img2)
    out = cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
    out = cv2.bitwise_and(out,src)
    show("Threshold",img1)
    show("Adaptive",img2)
    show("And",img3)
    show("Image",out)

gosei()

cv2.createTrackbar("Threshold","Threshold",thres,255, on_changeX)
cv2.createTrackbar("Offset","Adaptive",offset,10,on_changeY)

show("Threshold",img1)
show("Adaptive",img2)
show("And",img3)
show("Image",out)

cv2.waitKey(0)
cv2.destroyAllWindows()
