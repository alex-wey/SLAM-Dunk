import numpy as np
import cv2
from matplotlib import pyplot
from skimage.color import rgb2gray

#Input: two images
#Output: two (n, 2) matrix matches
def matchFeatures():
    # this is using SIFT

    ratio = 0.5
    # ignore the two lines below; i just used them to test my function 
    img1 = cv2.imread('wean_wide_interesting.left-rectified.00000600.t_001268594688.921905.png')
    img2 = cv2.imread('wean_wide_interesting.left-rectified.00000601.t_001268594688.992430.png')

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
            pointl1, pointr1 = sift.detectAndCompute(gray_img1, None)[0][left.queryIdx].pt
            pointl2, pointr2 = sift.detectAndCompute(gray_img2, None)[0][right.trainIdx].pt
            np.append(left_matches, [pointl1, pointr1])
            np.append(right_matches, [pointl2, pointr2])

            
# 682 points 
    print(left_matches)
    print(right_matches.shape)
 


    return left_matches, right_matches




matchFeatures()

