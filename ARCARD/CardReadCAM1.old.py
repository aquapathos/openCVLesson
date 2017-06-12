#-------------------------------------------------------------------------------
# Name:        CardReadCAM.py
# Purpose:
#
# Author:      dan
#
# Created:     09/04/2011
# Copyright:   (c) dan 2011
#-------------------------------------------------------------------------------
# coding: utf-8

import cv, cv2
import numpy as np
from math import *
from time import time

THRESHOLD = 0.5
ATHRESHOLD = 0.3
FOUNDTH = 7
BIGNUMBER = 1e6 # 1000000 （大きい数値）
WIDTHMIN = 128
HEIGHTMIN = 128

cardfilename = ["ar1.jpg","ar2.jpg","ar3.jpg","ar4.jpg","ar5.jpg","ar6.jpg"]

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
#flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)

def match_flann(desc1, desc2, a_threshold,r_threshold = 0.6):
    if len(desc1) == 0 or len(desc2)==0 :
        return BIGNUMBER,[]
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask1 = dist[:,0] < a_threshold
    mask2 = dist[:,0] / dist[:,1] < r_threshold # 特徴ごとのについて、第一候補との差と対象と第２候補との差の比がしきい値より小さいかどうかの配列
    mask = mask1 * mask2
    idx1 = np.arange(len(desc1))  # 0,1,2,3...
    pairs = np.int32( zip(idx1, idx2[:,0])) # for example, zip([0,1,2...],[a,b,c...]) => [(0,a),(1,b),...]
    return np.sum((dist[:,0][mask])), (pairs[mask])

# 対応点の集合から変換行列を求める
def locatePlanarObject(pairs,kps1,kps2,cn1):
    sn = len(pairs)
    cn2=[(0,0)]*4
    if sn < 4:
        return cn2
    srcpts = np.zeros((sn,2),np.float)
    dstpts = np.zeros((sn,2),np.float)
    for i in range(sn):
        (k1,k2) = pairs[i]
        pt1 = kps1[k1].pt
        pt2 = kps2[k2].pt
        srcpts[i] = pt1
        dstpts[i] = pt2
    h,_ = cv2.findHomography(srcpts,dstpts,cv2.RANSAC)

    for i in range(4):
        (x,y) = cn1[i]
        Z = (h[2,0]*x+h[2,1]*y+h[2,2])
        if abs(Z)<1e-6 :
            return [(0,0)]*4
        X = (h[0,0]*x+h[0,1]*y+h[0,2])/Z
        Y = (h[1,0]*x+h[1,1]*y+h[1,2])/Z
        cn2[i]=(int(X),int(Y))

    return cn2

# メイン関数
def main():
    # start = time()
    # 画像の読み込み
    MaxCheight = MaxCwidth = 0 # 一番大きい画像のサイズ
    ncards = len(cardfilename) # 認識対象カード数
    cardGrayImg = [None]*ncards # ぼかしたグレイ画像を保存するための配列
    cardColImg = [None]*ncards # 原画像を保存するための配列

    for i in range(ncards):
        filename = cardfilename[i]
        # ガウスフィルタでぼかす
        cardGrayImg[i] = cv2.GaussianBlur(cv2.imread(filename,0),(13,13),0)
        # 原画像
        cardColImg[i] = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)

        # 一番大きい画像のサイズを調べる
        h,w,_ = np.array(cardColImg[i]).shape
        if h > MaxCheight:
            MaxCheight = h
        if w > MaxCwidth:
            MaxCwidth = w

    # カメラの取り込み準備
    capture = cv2.VideoCapture(2)  # カム入力の場合
#    capture = cv2.VideoCapture("dummy.avi") # 動画入力の場合
    _,camColImg = capture.read()
#    camColImg = cv2.imread("ar1.jpg") # dummy
    camH,camW,_ = np.array(camColImg).shape
    camGrayImg = np.zeros((camH, camW), np.uint8)

    if camH > MaxCheight :
        Oheight = camH
    else:
        Oheight = MaxCheight
    Owidth = camW + MaxCwidth

    # 結果表示用画像（カムの画像＋認識結果の表示）
    mimg = np.zeros((Oheight,Owidth,3),np.uint8)

    # 元画像のSURF特徴の計算
    cardKey = [0]*ncards
    cardDpt = [0]*ncards

    surf = cv2.SURF(400,4,2,True,False)
    for i in range(ncards): # カードのSURF特徴を計算しておく
        (cardKey[i], cardDpt[i]) = surf.detect(cardGrayImg[i],None,False)
        cardDpt[i].shape = (-1, surf.descriptorSize())
        print i, u"特徴数は", len(cardKey[i])

    found = False
    wk = -1
    watching = 0 # 仮の候補を0番の画像とする

    # ROIの左上、右下の座標の初期値　カメラ画像の半分サイズで中央
    minx,miny = camW/4,camH/4
    maxx,maxy = minx+camW/2, miny+camH/2

    while wk != 27:
        start = time()

        # カメラからの画像取得
        tf = False
        while tf == False:
            try: tf,camColImg = capture.read()
            except: break
#        camColImg = cv2.imread("ar1.jpg") # dummy
        camGrayImg = cv2.cvtColor(camColImg, cv2.COLOR_BGR2GRAY)

        # 検索対象画像の特徴抽出
        # 特徴抽出対象領域の決定
        (sx1,sy1,sx2,sy2) = searcharea(minx,miny,maxx,maxy,camH,camW)
        # キャプチャ画像のSURF特徴を抽出する
        try:(camKey, camDpt) = surf.detect(camGrayImg[sy1:sy2,sx1:sx2],None,False)
        except: print sy1,sy2,sx1,sx2
        if len(camDpt) > 0:
            camDpt.shape = (-1, surf.descriptorSize())

        # 特徴間のマッチング
        cn,valmin,bestpairs = -1,BIGNUMBER,[]  # cn 候補番号, valmin 誤差の最小値
        if found == True: # 追跡状態 → watching 番だけをマッチング
            valmin, bestpairs = match_flann(cardDpt[watching], camDpt, ATHRESHOLD, THRESHOLD)
            if len(bestpairs) < FOUNDTH-2:
                found = False # lost
            else :
                cn = watching
        if found == False :# ロスト状態 → すべてマッチングして類似度最大のものを探す
            for i in range(ncards):
                 val, ptpairs = match_flann(cardDpt[i],camDpt, ATHRESHOLD, THRESHOLD)
                 if len(ptpairs) >= FOUNDTH and len(ptpairs)/val > len(bestpairs)/valmin :
                    bestpairs,cn,valmin = ptpairs,i,val
        watching = cn
        print "Best is %d, %d pairs with sum of Error %f" % (cn, len(bestpairs),valmin)

        # 結果画像のベース
        mimg[0:Oheight,0:MaxCwidth]=np.zeros((Oheight,MaxCwidth,3), np.uint8)
        mimg[Oheight-camH:Oheight,Owidth-camW:Owidth] = camColImg # CAM画像を右下に
        if watching >= 0:
            h,w,_=np.array(cardColImg[watching]).shape
            mimg[0:h,0:w] = cardColImg[watching] # カードの画像を左上に表示

            # print time()-start
            # 対応の描画
            for (i,nn) in bestpairs:
                (x1,y1)=cardKey[cn][i].pt
                (x2,y2)=camKey[nn].pt
                pt1 = (np.int(x1),np.int(y1))
                pt2 = (np.int(MaxCwidth+sx1+x2),np.int(Oheight-camH+sy1+y2))
                cv2.line(mimg,pt1,pt2,(0,255,255))
                cv2.rectangle(mimg,(sx1+MaxCwidth,sy1),(sx2+MaxCwidth,sy2),(255,255,255),1,8,0)

        nfound = len(bestpairs)
        if(nfound>FOUNDTH):
            # 対応点がFOUNDTH以上であれば発見候補
            # 検出領域を赤で描画
            (h00,w00)=cardGrayImg[cn].shape
            src_corners = [(0,0),(w00,0),(w00,h00),(0,h00)]
            pairs = []
            for i in range(nfound):
                [d1,d2]=bestpairs[i]
                pairs.append((d1,d2))
            xx,yy=pairs[0]
            corners4 = locatePlanarObject(pairs,cardKey[cn],camKey,src_corners)
            if found :
                ztmp=np.array(0.5*np.array(corners4)+0.5*np.array(dst_corners)) # 直前の4隅
                dst_corners = zip(ztmp[:,0],ztmp[:,1])
                # 矩形を少しでも安定させたいので、前回のコーナと計算値との中間点をコーナとして採用する
                # zip[[1,2],[2,3],[4,5],[4,3]] => [(1,2),(2,3),(4,5),(4,3)]
            else:
                dst_corners = corners4
            # 枠の描画 4隅が縮退していないときだけ描画

            if(len(set(dst_corners))==4):
                print "FOUND!",nfound,val
                found = True # 対応領域を見つけたと確信
                minx = miny = BIGNUMBER
                maxx = maxy = -BIGNUMBER
                for i in range(4):
                    (x1,y1)=dst_corners[i]
                    (x2,y2)=dst_corners[(i+1)%4]
                    x1,x2,y1,y2 = x1+sx1,x2+sx1,y1+sy1,y2+sy1 # (sx1,sy1)ROIの始点
                    if not(x1<0 or x2<0 or x1>=camW or x2>=camW
                        or y1<0 or y2<0 or y1>=camH or y2>=camH):
                        cv2.line(mimg,(np.int(x1+MaxCwidth),np.int(Oheight-camH+y1)),
                            (np.int(x2+MaxCwidth),np.int(Oheight-camH+y2)),(0,0,255),3)
                    minx = min(x1,x2,minx)
                    miny = min(y1,y2,miny)
                    maxx = max(x1,x2,maxx)
                    maxy = max(y1,y2,maxy)
            else: # 矩形が描けないようなときはロスト扱いだが ROI はそのまま
                found = False
                print "LOST"
        else:
            minx,miny = camW/4,camH/4
            maxx,maxy = minx+camW/2, miny+camH/2
            print "LOST",nfound,val
            found = False

        # 結果の表示
        cv2.imshow("Keypoint Matching",mimg)
        print u"所要時間",time()-start
        wk = cv2.waitKey(10)
#        cv2.destroyAllWindows()

def searcharea(minx,miny,maxx,maxy,camH,camW):
    wid = (maxx-minx)*0.6 # 4コーナを囲む矩形の1.2倍サイズのウィンドウを設定
    hig = (maxy-miny)*0.6
    cx = (maxx+minx)/2
    cy = (maxy+miny)/2
    sx1,sy1,sx2,sy2 = cx-wid,cy-hig,cx+wid,cy+hig
    if sx2 - sx1 < WIDTHMIN: # 横幅が狭すぎる
        sx1 = (sx2 + sx1)/2 - WIDTHMIN/2
        sx2 = sx1 + WIDTHMIN
    if sy2 - sy1 < WIDTHMIN: # 縦幅が狭すぎる
        sy1 = (sy2 + sy1)/2 - HEIGHTMIN/2
        sy2 = sy1 + HEIGHTMIN
    if sx1 < 0:
        sx1 = 0
        sx2 = WIDTHMIN
    if sy1 < 0:
        sy1 = 0
        sy2 = HEIGHTMIN
    if sx2 > camW:
        sx2 = camW
        sx1 = camW-WIDTHMIN
    if sy2 > camH:
        sy2 = camH
        sy1 = camH-HEIGHTMIN
    return int(sx1),int(sy1),int(sx2),int(sy2)

if __name__ == '__main__':
    main()
