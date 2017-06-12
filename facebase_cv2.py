# CV2 バージョン
import cv2

imgfile = "scene.jpg" # 顔検出の対象画像名
img = cv2.imread(imgfile,cv2.IMREAD_COLOR)

# 正面顔検出器のロード

cascade_file = "data\haarcascades\haarcascade_frontalface_alt.xml"  # Windows
# cascade_file = "data/haarcascades/haarcascade_frontalface_alt.xml" # Mac, Linux
cascade = cv2.CascadeClassifier(cascade_file)
faces = cascade.detectMultiScale(img)

for (x,y,w,h) in faces:
  cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),3)

cv2.imshow("result",img)
cv2.waitKey(0)

cv2.destroyAllWindows()
