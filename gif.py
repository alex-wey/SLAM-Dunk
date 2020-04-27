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

    if not os.path.isdir('poses'):
        os.mkdir('poses')

    gif_coords = np.zeros((matrix.shape[0], matrix.shape[1]))

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_xlim3d(np.min(matrix[:,0])-10, np.max(matrix[:,0])+10)
    ax.set_ylim3d(np.min(matrix[:,1])-10, np.max(matrix[:,1])+10)
    ax.set_zlim3d(np.min(matrix[:,2])-10, np.max(matrix[:,2])+10)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    for i, coords in enumerate(matrix):

        gif_coords[i] = coords
        x = gif_coords[0:i+1,0]
        y = gif_coords[0:i+1,1]
        z = gif_coords[0:i+1,2]
        
        ax.scatter(x, y, z, c='#d62728')
        plt.savefig('poses/camera_pose_' + str(i) + '.png')

    image_dir = os.path.dirname(os.path.abspath(__file__))
    images = io.collection.ImageCollection(image_dir + '/poses/*.png')
    imageio.mimsave('SLAM.gif', images, duration=1/8)
