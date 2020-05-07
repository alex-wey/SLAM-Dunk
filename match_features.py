import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import itertools
from PIL import Image
import matplotlib.pyplot as plt

#Input: two images
#Output: two (n, 2) matrix matches
#$ pip install image_slicer
def match_features(imgl, imgr, imgln, imgrn):
    print("Matching Features...")

    itr = 0
    width1 = 240
    width2 = 320

    # converting color images into grayscale imgaes
    gray_imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    gray_imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    gray_imgln = cv2.cvtColor(imgln, cv2.COLOR_BGR2GRAY)
    gray_imgrn = cv2.cvtColor(imgrn, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    total_lmatches = []
    total_rmatches = []
    total_lmatchesn = []
    total_rmatchesn = []

    row, col = gray_imgl.shape

    #dividing each image into four sections
    for i in range(0,row, width1):
        for j in range(0, col, width2):
            chop1 = gray_imgl[i : i +width1, j:j+width2]
            chop2 = gray_imgr[i : i +width1, j:j+width2]
            chop3 = gray_imgln[i : i +width1, j:j+width2]
            chop4 = gray_imgrn[i : i +width1, j:j+width2]

            chop1 = np.array(chop1)
            chop2 = np.array(chop2)
            chop3 = np.array(chop3)
            chop4 = np.array(chop4)

            #extracting the keypoints and descriptors using SIFT
            k1, d1 = sift.detectAndCompute(chop1, None)
            k2, d2 = sift.detectAndCompute(chop2, None)
            k1n, d1n = sift.detectAndCompute(chop3, None)
            k2n, d2n = sift.detectAndCompute(chop4, None)

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            try:
                #matches between imgl and imgln
                matches1n = bf.match(d1,d1n)
                #matches between imgl and imgr
                matches2 = bf.match(d1,d2)
                #matches between imgl and imgrn
                matches2n = bf.match(d1,d2n)

                # sorting and extracting top 50 best matches using the distance matrix
                matches1n = sorted(matches1n, key=lambda x: x.distance)
                matches1n = matches1n[:50]

                matches2 = sorted(matches2, key=lambda x: x.distance)
                matches2 = matches2[:50]

                matches2n = sorted(matches2n, key=lambda x: x.distance)
                matches2n = matches2n[:50]
            except Exception as e:
                print("No match")
                return np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2)), np.zeros((1,2))

            #getting the xy coordinate of the matches
            left_matches = []
            right_matches = []
            left_matchesn =[]
            right_matchesn = []
            for match in matches1n:
                    x1, y1 = k1[match.queryIdx].pt
                    x1n, y1n = k1n[match.trainIdx].pt
                    for match2 in matches2:
                        if (match2.queryIdx == match.queryIdx):
                            x2, y2 = k2[match2.trainIdx].pt
                            for match2n in matches2n:
                                if (match2n.queryIdx == match.queryIdx):
                                    x2n, y2n = k2n[match2n.trainIdx].pt
                                    #matching keypoints of imgr 
                                    right_matchesn.append(add_pixels(itr, x2n, y2n))
                                    # matching keypoints of imgl
                                    left_matches.append(add_pixels(itr, x1, y1))
                                    #matching keypoints of imgln
                                    left_matchesn.append(add_pixels(itr, x1n, y1n))
                                    # matching keypoints of imgrn
                                    right_matches.append(add_pixels(itr, x2, y2))
                                    break
                            break                  
                    continue

            lmatch = np.array(left_matches)
            lmatchn = np.array(left_matchesn)
            rmatch = np.array(right_matches)
            rmatchn = np.array(right_matchesn) 

            #getting the matches of all four sections
            if(lmatch.shape[0] != 0):
                total_lmatches.append(lmatch)
                total_rmatches.append(rmatch)
                total_lmatchesn.append(lmatchn)
                total_rmatchesn.append(rmatchn)

                lmatches = np.array(total_lmatches)
                rmatches = np.array(total_rmatches)
                lmatchesn = np.array(total_lmatchesn)
                rmatchesn = np.array(total_rmatchesn)

            itr += 1

    # shape = (n,2) where n is the number of matches
    lmatches = np.concatenate(lmatches, axis = 0)
    rmatches = np.concatenate(rmatches, axis = 0)
    lmatchesn = np.concatenate(lmatchesn, axis = 0)
    rmatchesn = np.concatenate(rmatchesn, axis = 0)
    
    return lmatches, rmatches, lmatchesn, rmatchesn


# helper function that adds pixels to differnt sections of an image
def add_pixels(itr, x,y):
    if itr is 1:
        x = x +320
    elif itr is 2:
        y = y + 240
    elif itr is 3:
        x = x+320
        y = y+240
    else:
        x = x

    return [x,y]
