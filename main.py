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
from update_camera_pose import update_camera_pose
from update_camera_pose import ransac_F_Matrix
from triangulate import triangulate
from match_features import match_features



def main():
	# load images, which are stored in a folder named 'rectified' in the same directory as this file
	data_dir = os.path.dirname(os.path.abspath(__file__))
	ic = io.collection.ImageCollection(data_dir+'/rectified/*.png')
	half_idx = len(ic)//2
	#load a few frames for testing
	left_ic = ic[0:50]
	right_ic = ic[half_idx:half_idx+50]
	ic = None
	# left_ic = ic[:half_idx]
	# right_ic = ic[half_idx:]
	# images have shape (480, 640, 3)

	# initial camera pose
	C = np.diag(np.ones(4))
	for i in range(len(left_ic)-1):
		imgl1 = left_ic[i]
		imgr1 = right_ic[i]
		imgl2 = left_ic[i+1]
		imgr2 = right_ic[i+1]
		# display left and right images for verification purposes
		'''
		curr = [left_ic[i],right_ic[i]]
		io.imshow_collection(curr)
		plt.show()
		'''
		# frame1
		matchesl1, matchesr1, matchesl2, matchesr2 = match_features(imgl1, imgr1, imgl2, imgr2)
		F1, inliers_a1, inliers_b1 = ransac_F_Matrix(matchesl1, matchesr1)
		coords3d1 = triangulate(inliers_a1, inliers_b1)
		# frame2
		# matchesl2, matchesr2 = match_features(imgl2, imgr2)
		F2, inliers_a2, inliers_b2 = ransac_F_Matrix(matchesl2, matchesr2)
		coords3d2 = triangulate(inliers_a2, inliers_b2)
		
		C = update_camera_pose(coords3d1, coords3d2, C)
		print(C)



main()
