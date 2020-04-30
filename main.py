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
from gif import gif

# camera poses array
plot_C = np.zeros((6509, 3))

# initial camera pose
C = np.diag(np.ones(4))

def main():
	# load images, which are stored in a folder named 'rectified' in the same directory as this file
	data_dir = os.path.dirname(os.path.abspath(__file__))
	ic = io.collection.ImageCollection(data_dir+'/rectified/*.png')
	half_idx = len(ic)//2
	#load a few frames for testing
	left_ic = ic[0:200]
	right_ic = ic[half_idx:half_idx+200]
	ic = None
	# left_ic = ic[:half_idx]
	# right_ic = ic[half_idx:]
	# images have shape (480, 640, 3)
	csv_file = open("points.csv", mode='w')
	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	# initial camera pose
	C = np.diag(np.ones(4))
	for i in range(len(left_ic)-1):
		print("Iteration ", i, "/", len(left_ic)-1)
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
		matchesl1, matchesr1, matchesl2, matchesr2 = match_features(imgl1, imgr1, imgl2, imgr2)
		F, inliers_a1, inliers_b1, inliers_a2, inliers_b2,  = ransac_F_Matrix(matchesl1, matchesr1,matchesl2, matchesr2)

		# frame1
		# F1, inliers_a1, inliers_b1 = ransac_F_Matrix(matchesl1, matchesr1)
		coords3d1 = triangulate(inliers_a1, inliers_b1)
		# coords3d1 = triangulate(matchesl1, matchesr1)
		
		# frame2
		# F2, inliers_a2, inliers_b2 = ransac_F_Matrix(matchesl2, matchesr2)
		coords3d2 = triangulate(inliers_a2, inliers_b2)
		# coords3d2 = triangulate(matchesl2, matchesr2)
		
		C_new = update_camera_pose(coords3d1, coords3d2, C)
		rejection_threshold = 0.5 #meters
		if(np.linalg.norm(C_new[0:3,3]-C[0:3,3]) < rejection_threshold):
			C = C_new
		else:
			print("New pose rejected")
		
		plot_C[i] = C[0:3,3].T
		csv_writer.writerow(plot_C[i])
		print("")

	gif(plot_C[0:200])


main()
