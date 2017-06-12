# -*- coding: utf-8 -*-
import numpy as np
import cv2

help_message = '''SURF image match

USAGE: findobj.py [ <image1> <image2> ]
'''

def fdetect(fn1,fn2,method):
    img1g = cv2.imread(fn1, 0)  #　グレーで読み込み
    img1c = cv2.imread(fn1, 1)  #　カラーで読み込み
    img2g = cv2.imread(fn2, 0)  #　グレーで読み込み
    img2c = cv2.imread(fn2, 1)  #　カラーで読み込み

    #特徴検出器で特徴検出
#
# FAST
    if(method == 'FAST'):
        detector = cv2.FastFeatureDetector_create()
    elif(method == 'MSER'):
        detector = cv2.MSER_create()
    elif(method == 'AKAZE'):
        detector = cv2.AKAZE_create()
    elif(method == 'BRISK'):
        detector = cv2.BRISK_create()
    elif(method == 'KAZE'):
        detector = cv2.KAZE_create()
    elif(method == 'ORB'): # ORB (Oriented FAST and Rotated BRIEF)
        detector = cv2.ORB_create()
    elif(method == 'SBD'): # SimpleBlobDetector
        detector = cv2.SimpleBlobDetector_create()
    elif(method == 'SIFT'):
        detector = cv2.xfeatures2d.SIFT_create()
    elif(method == 'SURF'):
        detector = cv2.xfeatures2d.SURF_create()
    else :
        print("No such method")
        return

#    OpenCV 2.x.x の場合は以下の通り
#    detector = cv2.FeatureDetector_create('Fast')
#    detector = cv2.FeatureDetector_create('SURF')
#    detector = cv2.FeatureDetector_create('SIFT')
#    detector = cv2.FeatureDetector_create('ORB')
#    detector = cv2.FeatureDetector_create('FASTX')
#    detector = cv2.FeatureDetector_create('FAST')
#    detector = cv2.FeatureDetector_create('STAR')
#    detector = cv2.FeatureDetector_create('BRISK')
#    detector = cv2.FeatureDetector_create('MSER')
#    detector = cv2.FeatureDetector_create('GFTT')
#    detector = cv2.FeatureDetector_create('HARRIS')
#    detector = cv2.FeatureDetector_create('Dense')
#    detector = cv2.FeatureDetector_create('KAZE')    # 3.0 >
#    detector = cv2.FeatureDetector_create('AKAZE')   # 3.0 >

    kd1 = detector.detect(img1g)
    kd2 = detector.detect(img2g)
    #それぞれの画像内で見つかった特徴点の数
    print('img1 - {0:d} features, img2 - {1:d} features'.format(len(kd1), len(kd2)))

    #特徴量の計算
    if method in ('ORB','AKAZE','KAZE','BRISK','SIFT','SURF'):
        kp1,desc1 = detector.detectAndCompute(img1g,None)
        kp2,desc2 = detector.detectAndCompute(img2g,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1,desc2, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                print(m.distance,"-",n.distance)
                good.append([m])

    #特徴の照合器を生成し、対応点を集めてくる
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    # matches = bf.match(desc1,desc2)

    '''
    matcher = cv2.BFMatcher()# Brute-Force Matcher生成
    if method in ('AKAZE', 'ORB', 'BRISK'):
        matches = matcher.knnMatch(desc1,desc2,k=2)
    #   matches = matcher.match(desc1,desc2)
    elif method in ('SIFT','KAZE'):
        matches = matcher.knnMatch(desc1,desc2,2)
    '''
    #対応
    distances = []
    for i,j in matches:
        distances.append([i.distance,j.distance])

    std = np.std(distances)
    print('{} matches'.format(len(matches)))
    print('min {0:.3f}'.format(min(distances))),
    print('max {0:.3f}'.format(max(distances))),
    print('std {0:.3f}'.format(std))
    d_thres = 0.5*(min(distances) + max(distances))-1*std
    # (平均-1標準偏差)をしきい値としてそれ以上差のある対応は無視することにする
    selects = [i for i in matches if i.distance < d_thres]
    print('{0:d} select matches'.format(len(selects)))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = np.array((img2.shape[0],img1.shape[1]+img2.shape[1],3),np.uint8)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)

    plt.imshow(img3),plt.show()

    # 上位２０までのマッチングを描画
    out = cv2.drawMatches(img1c,kp1,img2c,kp2,matches[:20],None,flags=2)
    '''
    以前は drawMatches という関数がなかったので、下のような描画プログラムで描画した
    h1, w1 = img1g.shape[:2]
    h2, w2 = img2g.shape[:2]

    #結果画像の作成
    out = np.zeros((max(h1,h2),w1+w2,3),np.uint8)
    out[:h2,w1:,] = img2c
    out[:h1,:w1,] = img1c

    for m in selects:
        matched_p1 = (int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1]))
        matched_p2 = (w1+int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1]))
        color = tuple([np.random.randint(0,255) for _ in range(3)])
        cv2.line(out,matched_p1,matched_p2,color)
    '''
    cv2.imshow('i3',out)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = 'box.png'
#        fn2 = 'box2.png'
        fn2 = 'box_in_scene.png'
        print(help_message)
    fdetect(fn1,fn2,'SIFT')
