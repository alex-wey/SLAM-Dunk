import numpy as np
import cv2
from matplotlib import pyplot
from skimage.color import rgb2gray

#Input: two images
#Output: two (n, 2) matrix matches
def match_features(img1, img2):

    print('Matching features...')

    ratio = 0.5

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(sift.detectAndCompute(gray_img1, None)[1], 
    sift.detectAndCompute(gray_img2, None)[1], k=2)
    
    left_matches = []
    right_matches = []
    counter = 0
    for left, right in matches:
        if left.distance < (right.distance * 0.7) : 
            matchl = np.zeros((1,2))
            matchr = np.zeros((1,2))
            pointl1, pointr1 = sift.detectAndCompute(gray_img1, None)[0][left.queryIdx].pt
            pointl2, pointr2 = sift.detectAndCompute(gray_img2, None)[0][right.trainIdx].pt
            matchl = [pointl1, pointr1]
            matchr = [pointl2, pointr2]
            left_matches.append(matchl)
            right_matches.append(matchr)
            counter += 1
    
    lmatches = np.array(left_matches)
    rmatches = np.array(right_matches)

    return lmatches, rmatches