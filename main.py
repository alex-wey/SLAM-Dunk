import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import os
import cv2
import argparse
from skimage import io 
from scipy import misc
from matplotlib import pyplot as plt
from update_camera_pose import updateCameraPose
from triangulate import triangulate
from match_features import matchFeatures

# initial camera pose
C = [1,1,1]

def main():
	# load images, which are stored in a folder named 'rectified' in the same directory as this file
	data_dir = os.path.dirname(os.path.abspath(__file__))
	ic = io.collection.ImageCollection(data_dir+'/rectified/*.png')
	left_ic = ic[:6509]
	right_ic = ic[6510:]
	# images have shape (480, 640, 3)
	for i in range(len(left_ic)):
		img1 = left_ic[i]
		img2 = right_ic[i]
		# display left and right images for verification purposes
		'''
		curr = [left_ic[i],right_ic[i]]
		io.imshow_collection(curr)
		plt.show()
		'''
		# frame1
		matchesl1, matchesr1 = matchFeatures(img1, img2)
		matchesl1, matchesr1 = RANSAC(matchesl1, matchesr1)
		coords3d1 = triangulate(matchesl1, matchesr1)
		# frame2
		matchesl2, matchesr2 = matchFeatures(img1, img2)
		matchesl2, matchesr2 = RANSAC(matchesl2, matchesr2)
		coords3d2 = triangulate(matchesl2, matchesr2)
		updateCameraPose(coords3d1, coords3d2)

main()