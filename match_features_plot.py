import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

#Input: two images
#Output: two (n, 2) matrix matches
def match_features(imgl, imgr, imgln, imgrn):
    #imgl, imgr = k-1 pairs  imgln, imgrn = k pairs
    print("Matching Features...")


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
    matches1n = bf.match(d1,d1n)
    #matches between imgl and imgr
    matches2 = bf.match(d1,d2)
    #matches between imgl and imgrn
    matches2n = bf.match(d1,d2n)


    matches1n = sorted(matches1n, key=lambda x: x.distance)
    matches1n = matches1n[:50]

    matches2 = sorted(matches2, key=lambda x: x.distance)
    matches2 = matches2[:50]

    matches2n = sorted(matches2n, key=lambda x: x.distance)
    matches2n = matches2n[:50]
    
    #get the same matches
    # same = new_matches & new_matches2
    
    #get the xy coordinate of the matches
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
                        left_matches.append([x1, y1])
                        left_matchesn.append([x1n, y1n])
                        right_matches.append([x2, y2])
                        right_matchesn.append([x2n, y2n])
                        break
                break                  
        continue


    lmatch = np.array(left_matches)
    lmatchn = np.array(left_matchesn)
    rmatch = np.array(right_matches)
    rmatchn = np.array(right_matchesn)

    checkimg1 = cv2.drawMatches(gray_imgl,k1,gray_imgr,k2,matches2, None, flags=2)
    checkimg2 = cv2.drawMatches(gray_imgl,k1,gray_imgrn,k2n,matches2n, None, flags=2)
    checkimg3 = cv2.drawMatches(gray_imgl,k1,gray_imgln,k1n,matches1n, None, flags=2)


    plt.imshow(checkimg1),plt.show()
    plt.imshow(checkimg2),plt.show()
    plt.imshow(checkimg3),plt.show()


    return lmatch, rmatch, lmatchn, rmatchn


