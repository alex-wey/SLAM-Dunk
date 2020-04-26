import numpy as np
import cv2
from matplotlib import pyplot
from skimage.color import rgb2gray

#Input: two images
#Output: two (n, 2) matrix matches

    # 3D-2D match features
#     1) Capture two stereo image pairs Il;k-1, Ir;k-1 and Il;k, Ir;k
#     2) Extract and match features between Il;k-1 and Il;k
#     3) Triangulate matched features for each stereo pair
#     4) Compute Tk from 3-D features Xk1 and Xk
#     5) Concatenate transformation by computing Ck ¼ Ck1Tk
#     6) Repeat from 1).

def match_features(imgl, imgr, imgln, imgrn):
    #imgl, imgr = k-1 pairs  imgln, imgrn = k pairs

    gray_imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    gray_imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    gray_imgln = cv2.cvtColor(imgln, cv2.COLOR_BGR2GRAY)
    gray_imgrn = cv2.cvtColor(imgrn, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    k1, d1 = sift.detectAndCompute(gray_imgl, None)
    k2, d2 = sift.detectAndCompute(gray_imgr, None)
    k1n, d1n = sift.detectAndCompute(gray_imgln, None)
    k2n, d2n = sift.detectAndCompute(gray_imgrn, None)


    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    #matches between imgl and imgln
    matches = bf.match(d1,d1n)
    #matches between imgr and imgr1
    matches2 = bf.match(d2, d2n)


    matches = sorted(matches, key=lambda x: x.distance)
    new_matches = matches[:50]

    matches2 = sorted(matches2, key=lambda x: x.distance)
    new_matches2 = matches2[:50]
    
    #get the same matches
    # same = new_matches & new_matches2
    
    #get the xy coordinate of the matches
    left_matches = []
    right_matches = []
    left_matchesn =[]
    right_matchesn = []
    for match in new_matches:
        x1, y1 = k1[match.queryIdx].pt
        x2, y2 = k1n[match.trainIdx].pt
        left_matches.append([x1, y1])
        left_matchesn.append([x2, y2])

    for match2 in new_matches2:
        x1n, y1n = k2[match2.queryIdx].pt
        x2n, y2n = k2n[match2.trainIdx].pt
        right_matches.append([x1n, y1n])
        right_matchesn.append([x2n,y2n])

    
    # lmatch = xy in k-1 left, lmatchn = xy in k left, 
    # rmatch = xy in k-1 right, rmatchn = xy in k right
    lmatch = np.array(left_matches)
    lmatchn = np.array(left_matchesn)
    rmatch = np.array(right_matches)
    rmatchn = np.array(right_matchesn)

    return lmatch, rmatch, lmatchn, rmatchn
