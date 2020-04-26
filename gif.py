import os
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Not used but needed to make 3D plots

# Input: numpy array of x,y,z coordinates of camera
# Output: 3D representation of a camera's path as gif
def gif(matrix):

    gif_coords = np.zeros((matrix.shape[0], matrix.shape[1]))

    for i, coords in enumerate(matrix):
        
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.set_zlim3d(0, 35)
        ax.set_ylim3d(0, 35)
        ax.set_xlim3d(0, 35)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        gif_coords[i] = coords
        x, y, z = np.hsplit(gif_coords[0:i+1], 3)
        ax.scatter(x, y, z)
        plt.savefig('poses/camera_pose_' + str(i) + '.png')

    image_dir = os.path.dirname(os.path.abspath(__file__))
    images = io.collection.ImageCollection(image_dir + '/poses/*.png')
    imageio.mimsave('SLAM.gif', images, duration=1/8)

def main():

    coords = np.zeros((30, 3))
    c = np.array([1, 2, 3])
    for i in range(30):
        coords[i] = c
        c = np.add(c, 1)

    gif(coords)

if __name__ == '__main__':
    main()
