import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import os
import cv2
import csv
from skimage import io 
from update_camera_pose import update_camera_pose
from update_camera_pose import ransac_F_Matrix
from triangulate import triangulate
from match_features import match_features
from match_features_plot import match_features_plot
from gif import gif
import argparse
import copy
import math

# camera poses array
plot_C = np.zeros((6509, 3))

# initial camera pose
C = np.diag(np.ones(4))

def main():

	#command line argument
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--plot", default = "no_img", help = "Either no_img, or plot_img" )
	args = parser.parse_args()
	PLOT = args.plot

	# load images, which are stored in a folder named 'rectified' in the same directory as this file
	data_dir = os.path.dirname(os.path.abspath(__file__))
	ic = io.collection.ImageCollection(data_dir+'/rectified/*.png')
	half_idx = len(ic)//2
	#load a few frames for testing
	'''
	left_ic = ic[0+513:500+513]
	right_ic = ic[half_idx+513:half_idx+500+513]
	ic = None
	'''
	left_ic = ic[:half_idx]
	right_ic = ic[half_idx:]
	# images have shape (480, 640, 3)
	csv_file = open("points.csv", mode='w')
	csv_features_file = open("features.csv", mode='w')
	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	csv_feature_writer = csv.writer(csv_features_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	# initial camera pose
	C = np.diag(np.ones(4))
	# store the last valid frames in case of skipping frames
	prevl = []
	prevr = []
	pts = np.array([])
	for i in range(len(left_ic)-1):
		print("Iteration ", i, "/", len(left_ic)-1)
		# current frame
		imgl1 = left_ic[i]
		imgr1 = right_ic[i]
		imgl2 = left_ic[i+1]
		imgr2 = right_ic[i+1]
		if PLOT.lower() == 'no_img' :
			matchesl1, matchesr1, matchesl2, matchesr2 = match_features(imgl1, imgr1, imgl2, imgr2)
		if PLOT.lower() == 'plot_img':
			matchesl1, matchesr1, matchesl2, matchesr2 = match_features_plot(imgl1, imgr1, imgl2, imgr2)
		if(matchesl1.shape[0] >= 3):	
			# get matches
			F, inliers_a1, inliers_b1, inliers_a2, inliers_b2,  = ransac_F_Matrix(matchesl1, matchesr1,matchesl2, matchesr2)
			std_threshold = 2
			# triangulate to find real-world points
			coords3d1 = triangulate(inliers_a1, inliers_b1)
			coords3d2 = triangulate(inliers_a2, inliers_b2)
			if(len(coords3d1)==0 or len(coords3d2)==0):
				print("skipping frame \n")
				continue
			z1 = coords3d1[:,2]
			z2 = coords3d2[:,2]
			if(i==0):
				pts = np.append(pts, z1)
			pts = np.append(pts, z2)
			if(len(pts)>100):
				pts = pts[:100]
			mean = np.mean(pts)
			std = np.std(pts)

			mean1 = np.mean(z1)
			mean2 = np.mean(z2)

			zstd1 = mean1/std
			zstd2 = mean2/std

			print("std:", zstd1, zstd2)
			# both frames k and k+1 invalid, skip
			if(zstd1>=std_threshold and zstd2>=std_threshold):
				print("skipping frame \n")
				continue
			# frame k invalid, frame k+1 valid
			elif(zstd1>=std_threshold and zstd2<std_threshold):
				# use last valid frame if exists, otherwise skip
				if(prevl != [] and prevr != []):
					matchesl1, matchesr1, matchesl2, matchesr2 = match_features(prevl, prevr, imgl2, imgr2)
					F, inliers_a1, inliers_b1, inliers_a2, inliers_b2,  = ransac_F_Matrix(matchesl1, matchesr1,matchesl2, matchesr2)
					coords3d1 = triangulate(inliers_a1, inliers_b1)
					coords3d2 = triangulate(inliers_a2, inliers_b2)
				else:
					print("skipping frame \n")
					continue
			# frame k valid, frame k+1 invalid
			elif(zstd1<std_threshold and zstd2>=std_threshold):
				prevl = copy.deepcopy(imgl1)
				prevr = copy.deepcopy(imgr1)
				print("skipping frame \n")
				continue
			# both frames valid
			else:
				prevl = []
				prevr = []
			
			inliers = coords3d1.shape[0]
			coords_abs = C.T @ np.append(coords3d1, np.ones((inliers, 1)), axis=1).T
			csv_feature_writer.writerows(coords_abs[0:3,:].T)
			C_new = update_camera_pose(coords3d1, coords3d2, C)
			pose_distance = np.linalg.norm(C_new[0:3,3] - C[0:3,3])
			rejection_threshold = 0.5 #meters
			print("Pose", C[0:3,3])
			print("Pose Distance", pose_distance)
			if(pose_distance < rejection_threshold):
				C = C_new
			else:
				print("pose rejected")

		plot_C[i] = C[0:3,3].T
		csv_writer.writerow(plot_C[i])
		print("")

	gif(plot_C)


main()
