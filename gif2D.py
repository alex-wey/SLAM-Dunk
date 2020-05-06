import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
!pip install ffmpeg-python

# Input: numpy array of x,y,z coordinates of camera
# Output: 3D representation of a camera's path as gif
def gif(matrix):
    # instantiate writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

    # plot actual points
    actual_points = pd.read_csv("actual_points.csv").values
    x_actual = actual_points[:,0]
    z_actual = actual_points[:,1]

    # set up graphs
    df = pd.DataFrame(matrix, columns=["x","y"])
    frn = len(df)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_actual, z_actual, c='b')
    sc = ax.scatter([],[], c='r', alpha=0.5)
    
    # label axis
    ax.set_title('Camera Pose')
    ax.set_xlim(-10, 100)
    ax.set_xlabel('x')
    ax.set_ylim(-45, 50)
    ax.set_ylabel('z')

    def animate(i):
        data = np.hstack((df.x.values[:i, np.newaxis], df.y.values[:i, np.newaxis]))
        sc.set_offsets(data)

    # animate and save
    ani = animation.FuncAnimation(fig, animate, frames=frn, interval=5)
    ani.save('SLAM_2D.mp4', writer=writer)
