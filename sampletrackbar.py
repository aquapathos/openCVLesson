# -*- coding: utf-8 -*-
import cv2

def main():
    thres = 128

    image = cv2.imread("lena.jpg",1)

    def on_change(thres):
        binarization(thres)

    def binarization(thres):
        global gray
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray,thres,255,cv2.THRESH_BINARY)
        cv2.imshow("Binarization",gray)

    cv2.imshow("Source",image)
    binarization(thres)
    cv2.imshow("Binarization",gray)
    cv2.createTrackbar("Threshold","Binarization",thres,255, on_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
